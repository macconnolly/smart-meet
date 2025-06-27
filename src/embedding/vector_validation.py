"""
Vector validation utilities for cognitive vectors.

This module provides comprehensive validation for 400D cognitive vectors,
including range checks, dimension validation, and statistical analysis.
"""

import logging
from typing import List, Tuple, Dict, Optional, Any
import numpy as np
from dataclasses import dataclass

from ..models.entities import Vector
from .vector_manager import VectorManager, get_vector_manager

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of vector validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    stats: Optional[Dict[str, Any]] = None


class VectorValidator:
    """
    Comprehensive validator for cognitive vectors.
    
    Performs validation checks:
    - Dimension correctness (384D + 16D = 400D)
    - Semantic normalization
    - Cognitive dimension ranges [0, 1]
    - NaN/inf detection
    - Statistical anomalies
    """
    
    def __init__(self, vector_manager: Optional[VectorManager] = None):
        """
        Initialize the validator.
        
        Args:
            vector_manager: VectorManager instance (uses singleton if None)
        """
        self.vector_manager = vector_manager or get_vector_manager()
        
        # Thresholds for anomaly detection
        self.min_semantic_norm = 0.99
        self.max_semantic_norm = 1.01
        self.min_dimension_std = 0.05  # Too uniform is suspicious
        self.max_dimension_zeros = 10   # Max number of zero dimensions
        
    def validate_vector(
        self,
        vector: Vector,
        strict: bool = True
    ) -> ValidationResult:
        """
        Validate a single vector.
        
        Args:
            vector: Vector to validate
            strict: Whether to apply strict validation rules
            
        Returns:
            ValidationResult with errors and warnings
        """
        errors = []
        warnings = []
        
        # Basic validation using VectorManager
        is_valid, error_msg = self.vector_manager.validate_vector(vector)
        if not is_valid:
            errors.append(error_msg)
        
        # Additional checks
        stats = self._compute_stats(vector)
        
        # Check semantic normalization (strict)
        if strict and (stats['semantic_norm'] < self.min_semantic_norm or 
                      stats['semantic_norm'] > self.max_semantic_norm):
            errors.append(
                f"Semantic vector not properly normalized: norm={stats['semantic_norm']:.4f}"
            )
        
        # Check for too many zero dimensions
        zero_dims = np.sum(vector.dimensions == 0)
        if zero_dims > self.max_dimension_zeros:
            warnings.append(f"Too many zero dimensions: {zero_dims}/{len(vector.dimensions)}")
        
        # Check for low variance (all dimensions similar)
        if stats['dimensions_std'] < self.min_dimension_std:
            warnings.append(
                f"Low dimension variance: std={stats['dimensions_std']:.4f} "
                "(dimensions too uniform)"
            )
        
        # Check for extreme values
        if stats['dimensions_min'] == 0 and stats['dimensions_max'] == 1:
            extreme_count = np.sum((vector.dimensions == 0) | (vector.dimensions == 1))
            if extreme_count > 8:  # Half of dimensions
                warnings.append(
                    f"Many extreme dimension values: {extreme_count} dimensions at 0 or 1"
                )
        
        # Final validation result
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            stats=stats
        )
    
    def validate_batch(
        self,
        vectors: List[Vector],
        strict: bool = True,
        fail_fast: bool = False
    ) -> List[ValidationResult]:
        """
        Validate multiple vectors.
        
        Args:
            vectors: List of vectors to validate
            strict: Whether to apply strict validation rules
            fail_fast: Stop on first invalid vector
            
        Returns:
            List of ValidationResult objects
        """
        results = []
        
        for i, vector in enumerate(vectors):
            result = self.validate_vector(vector, strict=strict)
            results.append(result)
            
            if fail_fast and not result.is_valid:
                logger.error(f"Validation failed at vector {i}: {result.errors}")
                break
        
        return results
    
    def _compute_stats(self, vector: Vector) -> Dict[str, Any]:
        """Compute detailed statistics for a vector."""
        stats = self.vector_manager.get_vector_stats(vector)
        
        # Add additional stats
        return {
            'semantic_norm': stats.semantic_norm,
            'dimensions_mean': stats.dimensions_mean,
            'dimensions_std': stats.dimensions_std,
            'dimensions_min': stats.dimensions_min,
            'dimensions_max': stats.dimensions_max,
            'zero_count': int(np.sum(vector.dimensions == 0)),
            'one_count': int(np.sum(vector.dimensions == 1)),
            'unique_values': int(len(np.unique(vector.dimensions))),
            'entropy': float(self._compute_entropy(vector.dimensions))
        }
    
    def _compute_entropy(self, dimensions: np.ndarray) -> float:
        """Compute entropy of dimension values."""
        # Discretize to 10 bins
        hist, _ = np.histogram(dimensions, bins=10, range=(0, 1))
        hist = hist[hist > 0]  # Remove zero bins
        
        if len(hist) == 0:
            return 0.0
        
        # Normalize
        probs = hist / np.sum(hist)
        
        # Compute entropy
        entropy = -np.sum(probs * np.log2(probs))
        
        return entropy
    
    def check_vector_similarity(
        self,
        vector1: Vector,
        vector2: Vector,
        threshold: float = 0.95
    ) -> Tuple[bool, float]:
        """
        Check if two vectors are too similar (potential duplicates).
        
        Args:
            vector1: First vector
            vector2: Second vector
            threshold: Similarity threshold for flagging
            
        Returns:
            Tuple of (is_too_similar, similarity_score)
        """
        similarity = self.vector_manager.compute_similarity(vector1, vector2)
        is_too_similar = similarity > threshold
        
        return is_too_similar, similarity
    
    def find_anomalies(
        self,
        vectors: List[Vector],
        z_threshold: float = 3.0
    ) -> List[Tuple[int, str]]:
        """
        Find anomalous vectors in a batch.
        
        Args:
            vectors: List of vectors to analyze
            z_threshold: Z-score threshold for anomaly detection
            
        Returns:
            List of (index, reason) tuples for anomalous vectors
        """
        if len(vectors) < 3:
            return []  # Need at least 3 vectors for statistics
        
        anomalies = []
        
        # Collect dimension statistics
        all_dimensions = np.vstack([v.dimensions for v in vectors])
        dim_means = np.mean(all_dimensions, axis=0)
        dim_stds = np.std(all_dimensions, axis=0)
        
        # Check each vector
        for i, vector in enumerate(vectors):
            # Check for dimension outliers
            if dim_stds.any():
                z_scores = np.abs((vector.dimensions - dim_means) / (dim_stds + 1e-8))
                outlier_dims = np.where(z_scores > z_threshold)[0]
                
                if len(outlier_dims) > 0:
                    dim_names = self._get_dimension_names()
                    outlier_names = [dim_names[d] for d in outlier_dims]
                    anomalies.append((
                        i,
                        f"Outlier dimensions: {', '.join(outlier_names)}"
                    ))
            
            # Check for unusual patterns
            result = self.validate_vector(vector, strict=False)
            if result.warnings:
                anomalies.append((i, "; ".join(result.warnings)))
        
        return anomalies
    
    def _get_dimension_names(self) -> List[str]:
        """Get ordered list of dimension names."""
        return [
            # Temporal
            "urgency", "deadline_proximity", "sequence_position", "duration_relevance",
            # Emotional
            "polarity", "intensity", "confidence",
            # Social
            "authority", "influence", "team_dynamics",
            # Causal
            "dependencies", "impact", "risk_factors",
            # Evolutionary
            "change_rate", "innovation_level", "adaptation_need"
        ]
    
    def generate_validation_report(
        self,
        vectors: List[Vector]
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive validation report for a batch of vectors.
        
        Args:
            vectors: List of vectors to analyze
            
        Returns:
            Dictionary with validation statistics and issues
        """
        results = self.validate_batch(vectors, strict=True, fail_fast=False)
        
        # Count valid/invalid
        valid_count = sum(1 for r in results if r.is_valid)
        invalid_count = len(results) - valid_count
        
        # Collect all errors and warnings
        all_errors = []
        all_warnings = []
        for i, result in enumerate(results):
            for error in result.errors:
                all_errors.append(f"Vector {i}: {error}")
            for warning in result.warnings:
                all_warnings.append(f"Vector {i}: {warning}")
        
        # Find anomalies
        anomalies = self.find_anomalies(vectors) if len(vectors) >= 3 else []
        
        # Compute aggregate statistics
        if vectors:
            all_dimensions = np.vstack([v.dimensions for v in vectors])
            dim_stats = {
                name: {
                    "mean": float(np.mean(all_dimensions[:, i])),
                    "std": float(np.std(all_dimensions[:, i])),
                    "min": float(np.min(all_dimensions[:, i])),
                    "max": float(np.max(all_dimensions[:, i]))
                }
                for i, name in enumerate(self._get_dimension_names())
            }
        else:
            dim_stats = {}
        
        return {
            "total_vectors": len(vectors),
            "valid_count": valid_count,
            "invalid_count": invalid_count,
            "validation_rate": valid_count / len(vectors) if vectors else 0,
            "errors": all_errors[:10],  # Top 10 errors
            "warnings": all_warnings[:10],  # Top 10 warnings
            "anomalies": [
                {"index": idx, "reason": reason}
                for idx, reason in anomalies[:5]  # Top 5 anomalies
            ],
            "dimension_statistics": dim_stats,
            "error_count": len(all_errors),
            "warning_count": len(all_warnings)
        }


# Singleton instance
_validator_instance: Optional[VectorValidator] = None


def get_vector_validator() -> VectorValidator:
    """Get or create the global vector validator instance."""
    global _validator_instance
    
    if _validator_instance is None:
        _validator_instance = VectorValidator()
    
    return _validator_instance


# Example usage and testing
if __name__ == "__main__":
    # Create validator
    validator = get_vector_validator()
    manager = get_vector_manager()
    
    # Create test vectors
    vectors = []
    
    # Valid vector
    semantic1 = np.random.randn(384)
    semantic1 = semantic1 / np.linalg.norm(semantic1)
    cognitive1 = np.random.rand(16) * 0.8 + 0.1  # Range [0.1, 0.9]
    vector1 = manager.compose_vector(semantic1, cognitive1)
    vectors.append(vector1)
    
    # Invalid vector (not normalized)
    semantic2 = np.random.randn(384) * 2
    cognitive2 = np.random.rand(16)
    vector2 = Vector(semantic=semantic2, dimensions=cognitive2)
    vectors.append(vector2)
    
    # Suspicious vector (all dimensions similar)
    semantic3 = np.random.randn(384)
    semantic3 = semantic3 / np.linalg.norm(semantic3)
    cognitive3 = np.ones(16) * 0.5
    vector3 = manager.compose_vector(semantic3, cognitive3)
    vectors.append(vector3)
    
    # Validate individually
    for i, vector in enumerate(vectors):
        result = validator.validate_vector(vector)
        print(f"\nVector {i+1}:")
        print(f"  Valid: {result.is_valid}")
        print(f"  Errors: {result.errors}")
        print(f"  Warnings: {result.warnings}")
    
    # Generate report
    print("\nValidation Report:")
    report = validator.generate_validation_report(vectors)
    print(f"  Total: {report['total_vectors']}")
    print(f"  Valid: {report['valid_count']}")
    print(f"  Invalid: {report['invalid_count']}")
    print(f"  Errors: {report['error_count']}")
    print(f"  Warnings: {report['warning_count']}")
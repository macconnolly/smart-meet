"""
Enhanced cognitive encoder with multi-dimensional fusion layer.

This module extends the basic ONNX encoder to combine semantic embeddings
with cognitive dimensions through a learned fusion layer, producing rich
cognitive memory representations.

Reference: IMPLEMENTATION_GUIDE.md - Phase 2: Enhanced Encoding
"""

from typing import Any, Dict, List, Optional
import numpy as np
from loguru import logger
from pathlib import Path

from ...embedding.onnx_encoder import ONNXEncoder
from ...extraction.dimensions.dimension_analyzer import DimensionAnalyzer


class CognitiveFusionLayer:
    """
    Linear fusion layer that combines semantic and dimensional features.

    Takes concatenated Sentence-BERT embeddings and cognitive dimensions
    and transforms them into a unified cognitive representation
    through a simple linear transformation with configurable output dimensions.
    """

    def __init__(self, semantic_dim: int, cognitive_dim: int, output_dim: int) -> None:
        """
        Initialize the fusion layer.

        Args:
            semantic_dim: Dimensionality of semantic embeddings (384D from ONNX)
            cognitive_dim: Dimensionality of cognitive dimensions (16D)
            output_dim: Dimensionality of final cognitive embeddings (400D)
        """
        self.semantic_dim = semantic_dim
        self.cognitive_dim = cognitive_dim
        self.output_dim = output_dim
        self.input_dim = semantic_dim + cognitive_dim

        # Initialize weights and bias for linear transformation
        self.weight = np.random.normal(0, 0.1, (self.input_dim, output_dim)).astype(
            np.float32
        )
        self.bias = np.zeros(output_dim, dtype=np.float32)

        # Layer normalization parameters
        self.layer_norm_weight = np.ones(output_dim, dtype=np.float32)
        self.layer_norm_bias = np.zeros(output_dim, dtype=np.float32)
        self.layer_norm_eps = 1e-5

        # Initialize weights using Xavier uniform-like initialization
        self._initialize_weights()

        logger.debug(
            "Cognitive fusion layer initialized",
            semantic_dim=semantic_dim,
            cognitive_dim=cognitive_dim,
            output_dim=output_dim,
            total_params=self.weight.size
            + self.bias.size
            + self.layer_norm_weight.size
            + self.layer_norm_bias.size,
        )

    def _initialize_weights(self) -> None:
        """Initialize layer weights using Xavier uniform-like initialization."""
        # Xavier uniform initialization approximation
        limit = np.sqrt(6.0 / (self.input_dim + self.output_dim))
        self.weight = np.random.uniform(
            -limit, limit, (self.input_dim, self.output_dim)
        ).astype(np.float32)
        self.bias = np.zeros(self.output_dim, dtype=np.float32)

    def forward(
        self, semantic_embedding: np.ndarray, cognitive_dimensions: np.ndarray
    ) -> np.ndarray:
        """
        Forward pass through the fusion layer.

        Args:
            semantic_embedding: Sentence-BERT embedding array [batch_size, semantic_dim] or [semantic_dim]
            cognitive_dimensions: Cognitive dimensions array [batch_size, cognitive_dim] or [cognitive_dim]

        Returns:
            np.ndarray: Fused cognitive embedding [batch_size, output_dim] or [output_dim]
        """
        # Handle both single and batch inputs
        if semantic_embedding.ndim == 1:
            semantic_embedding = semantic_embedding.reshape(1, -1)
            single_input = True
        else:
            single_input = False

        if cognitive_dimensions.ndim == 1:
            cognitive_dimensions = cognitive_dimensions.reshape(1, -1)

        # Ensure batch dimensions match
        batch_size = semantic_embedding.shape[0]
        if cognitive_dimensions.shape[0] != batch_size:
            cognitive_dimensions = np.tile(cognitive_dimensions, (batch_size, 1))

        # Concatenate semantic and cognitive features
        combined_features = np.concatenate(
            [semantic_embedding, cognitive_dimensions], axis=1
        )

        # Apply linear transformation
        fused_embedding = np.dot(combined_features, self.weight) + self.bias

        # Apply layer normalization
        fused_embedding = self._layer_norm(fused_embedding)

        # Return single array if single input was provided
        if single_input:
            fused_embedding = fused_embedding.squeeze(0)

        return fused_embedding

    def _layer_norm(self, x: np.ndarray) -> np.ndarray:
        """Apply layer normalization."""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        normalized = (x - mean) / np.sqrt(var + self.layer_norm_eps)
        return normalized * self.layer_norm_weight + self.layer_norm_bias


class EnhancedCognitiveEncoder:
    """
    Enhanced cognitive encoding system combining semantic and dimensional analysis.

    This encoder integrates ONNX-based semantic embeddings with rule-based
    cognitive dimensions through a learned fusion layer to create rich
    cognitive memory representations suitable for the multi-layered memory system.
    
    Extends the basic ONNX encoder with cognitive dimension fusion.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        config_path: Optional[str] = None,
        fusion_weights_path: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the enhanced cognitive encoder.

        Args:
            model_path: Path to ONNX model file
            tokenizer_path: Path to tokenizer directory
            config_path: Path to model config JSON
            fusion_weights_path: Path to pre-trained fusion layer weights (NumPy format)
            cache_dir: Directory for caching embeddings
        """
        # Initialize base ONNX encoder
        self.semantic_encoder = ONNXEncoder(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            config_path=config_path,
            cache_dir=cache_dir
        )
        
        # Initialize dimension analyzer
        self.dimension_analyzer = DimensionAnalyzer()
        
        # Get dimensions for fusion layer
        self.semantic_dim = 384  # ONNX model output dimension
        self.cognitive_dim = 16  # Total cognitive dimensions
        self.output_dim = 400   # Final output dimension
        
        # Initialize fusion layer
        self.fusion_layer = CognitiveFusionLayer(
            semantic_dim=self.semantic_dim,
            cognitive_dim=self.cognitive_dim,
            output_dim=self.output_dim,
        )
        
        # Load pre-trained weights if provided
        if fusion_weights_path and Path(fusion_weights_path).exists():
            self.load_fusion_weights(fusion_weights_path)
        
        logger.info(
            "Enhanced cognitive encoder initialized",
            semantic_dim=self.semantic_dim,
            cognitive_dim=self.cognitive_dim,
            output_dim=self.output_dim,
        )

    def encode(self, text: str, context: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Encode text into a cognitive memory representation.

        Args:
            text: Input text to encode
            context: Optional context information for dimension extraction

        Returns:
            np.ndarray: Cognitive embedding with 400 dimensions
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for cognitive encoding")
            return np.zeros(self.output_dim, dtype=np.float32)

        try:
            # Extract semantic embedding (384D)
            semantic_embedding = self.semantic_encoder.encode(text)
            
            # Extract cognitive dimensions (16D)
            cognitive_dimensions = self.dimension_analyzer.extract_dimensions(text)
            
            # Fuse through linear layer (400D)
            cognitive_embedding = self.fusion_layer.forward(
                semantic_embedding, cognitive_dimensions
            )
            
            logger.debug(
                "Text encoded into cognitive representation",
                text_length=len(text),
                semantic_shape=semantic_embedding.shape,
                cognitive_dims_shape=cognitive_dimensions.shape,
                output_shape=cognitive_embedding.shape,
            )
            
            return cognitive_embedding

        except Exception as e:
            logger.error(
                "Failed to encode text cognitively",
                text_preview=text[:100] + "..." if len(text) > 100 else text,
                error=str(e),
            )
            return np.zeros(self.output_dim, dtype=np.float32)

    def encode_batch(
        self, texts: List[str], contexts: Optional[List[Dict[str, Any]]] = None
    ) -> np.ndarray:
        """
        Encode multiple texts into cognitive memory representations.

        Args:
            texts: List of input texts to encode
            contexts: Optional list of context information

        Returns:
            np.ndarray: Batch of cognitive embeddings [batch_size, 400]
        """
        if not texts:
            logger.warning("Empty text list provided for batch encoding")
            return np.zeros((0, self.output_dim), dtype=np.float32)

        try:
            # Extract semantic embeddings for all texts
            semantic_embeddings = self.semantic_encoder.encode_batch(texts)
            
            # Extract cognitive dimensions for all texts
            batch_cognitive_dims = []
            for i, text in enumerate(texts):
                context = contexts[i] if contexts and i < len(contexts) else None
                cognitive_dims = self.dimension_analyzer.extract_dimensions(text)
                batch_cognitive_dims.append(cognitive_dims)
            
            # Stack cognitive dimensions into batch array
            cognitive_dims_batch = np.stack(batch_cognitive_dims, axis=0)
            
            # Fuse through linear layer
            cognitive_embeddings = self.fusion_layer.forward(
                semantic_embeddings, cognitive_dims_batch
            )
            
            logger.debug(
                "Batch encoded into cognitive representations",
                batch_size=len(texts),
                semantic_shape=semantic_embeddings.shape,
                cognitive_dims_shape=cognitive_dims_batch.shape,
                output_shape=cognitive_embeddings.shape,
            )
            
            return cognitive_embeddings

        except Exception as e:
            logger.error(
                "Failed to encode text batch cognitively",
                batch_size=len(texts),
                error=str(e),
            )
            return np.zeros((len(texts), self.output_dim), dtype=np.float32)

    def get_dimension_breakdown(self, text: str) -> Dict[str, Any]:
        """
        Get detailed breakdown of dimensions extracted from text.

        Args:
            text: Input text to analyze

        Returns:
            dict: Detailed dimension analysis including scores and explanations
        """
        try:
            # Get semantic embedding
            semantic_embedding = self.semantic_encoder.encode(text)
            
            # Get cognitive dimensions with detailed breakdown
            cognitive_dims = self.dimension_analyzer.extract_dimensions(text)
            
            # Get individual dimension values
            temporal_dims = cognitive_dims[:4]
            emotional_dims = cognitive_dims[4:7]
            social_dims = cognitive_dims[7:10]
            causal_dims = cognitive_dims[10:13]
            strategic_dims = cognitive_dims[13:16]
            
            breakdown = {
                "semantic_embedding_norm": float(np.linalg.norm(semantic_embedding)),
                "dimensions": {
                    "temporal": {
                        "values": temporal_dims.tolist(),
                        "names": ["urgency", "deadline_proximity", "sequence_position", "duration_relevance"],
                        "total_activation": float(np.sum(temporal_dims)),
                    },
                    "emotional": {
                        "values": emotional_dims.tolist(),
                        "names": ["polarity", "intensity", "confidence"],
                        "total_activation": float(np.sum(emotional_dims)),
                    },
                    "social": {
                        "values": social_dims.tolist(),
                        "names": ["authority", "influence", "team_dynamics"],
                        "total_activation": float(np.sum(social_dims)),
                    },
                    "causal": {
                        "values": causal_dims.tolist(),
                        "names": ["dependencies", "impact", "risk_factors"],
                        "total_activation": float(np.sum(causal_dims)),
                    },
                    "strategic": {
                        "values": strategic_dims.tolist(),
                        "names": ["alignment", "priority", "business_impact"],
                        "total_activation": float(np.sum(strategic_dims)),
                    }
                }
            }
            
            return breakdown

        except Exception as e:
            logger.error(
                "Failed to generate dimension breakdown",
                text_preview=text[:100] + "..." if len(text) > 100 else text,
                error=str(e),
            )
            return {"error": str(e)}

    def save_fusion_weights(self, weights_path: str) -> bool:
        """Save current fusion layer weights (NumPy format)."""
        try:
            weights_data = {
                "weight": self.fusion_layer.weight,
                "bias": self.fusion_layer.bias,
                "layer_norm_weight": self.fusion_layer.layer_norm_weight,
                "layer_norm_bias": self.fusion_layer.layer_norm_bias,
            }
            np.savez(weights_path, **weights_data)
            logger.info("Fusion layer weights saved successfully", path=weights_path)
            return True
        except Exception as e:
            logger.error(
                "Failed to save fusion weights", path=weights_path, error=str(e)
            )
            return False

    def load_fusion_weights(self, weights_path: str) -> bool:
        """Load fusion layer weights (NumPy format)."""
        try:
            weights_data = np.load(weights_path)
            self.fusion_layer.weight = weights_data["weight"]
            self.fusion_layer.bias = weights_data["bias"]
            self.fusion_layer.layer_norm_weight = weights_data["layer_norm_weight"]
            self.fusion_layer.layer_norm_bias = weights_data["layer_norm_bias"]
            logger.info("Fusion layer weights loaded successfully", path=weights_path)
            return True
        except Exception as e:
            logger.warning(
                "Failed to load fusion weights, using random initialization",
                path=weights_path,
                error=str(e),
            )
            return False

    def get_encoder_info(self) -> Dict[str, Any]:
        """Get information about the encoder configuration."""
        return {
            "semantic_encoder": {
                "type": "ONNX",
                "dimension": self.semantic_dim,
            },
            "dimension_analyzer": {
                "total_dimensions": self.cognitive_dim,
                "categories": ["temporal", "emotional", "social", "causal", "strategic"]
            },
            "fusion_layer": {
                "input_dim": self.semantic_dim + self.cognitive_dim,
                "output_dim": self.output_dim,
                "parameters": self.fusion_layer.weight.size + self.fusion_layer.bias.size,
                "implementation": "NumPy linear layer with layer normalization",
            },
            "total_output_dimensions": self.output_dim
        }

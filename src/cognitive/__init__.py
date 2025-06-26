"""
Cognitive Processing Engines Package

This package contains the core cognitive processing engines that
provide intelligent memory retrieval, bridge discovery, and
consolidation capabilities.
"""

from .activation.engine import (
    BFSActivationEngine,
    ParallelActivationEngine,
    AdaptiveActivationEngine,
    ActivationConfig,
    ActivationPath,
)

from .bridges.engine import (
    DistanceInversionEngine,
    SerendipityEngine,
    CrossModalBridgeEngine,
    BridgeConfig,
    BridgeContext,
    BridgeScore,
)

from .consolidation.engine import (
    IntelligentConsolidationEngine,
    IncrementalConsolidationEngine,
    ConsolidationConfig,
    ConsolidationResult,
    ClusterQuality,
)

__all__ = [
    # Activation engines
    'BFSActivationEngine',
    'ParallelActivationEngine', 
    'AdaptiveActivationEngine',
    'ActivationConfig',
    'ActivationPath',
    
    # Bridge engines
    'DistanceInversionEngine',
    'SerendipityEngine',
    'CrossModalBridgeEngine',
    'BridgeConfig',
    'BridgeContext',
    'BridgeScore',
    
    # Consolidation engines
    'IntelligentConsolidationEngine',
    'IncrementalConsolidationEngine',
    'ConsolidationConfig',
    'ConsolidationResult',
    'ClusterQuality',
]

# @TODO: Package metadata
__version__ = '2.0.0'
__description__ = 'Cognitive processing engines for memory intelligence'

# @TODO: Engine registry for dynamic loading
ENGINE_REGISTRY = {
    'activation': {
        'bfs': BFSActivationEngine,
        'parallel': ParallelActivationEngine,
        'adaptive': AdaptiveActivationEngine,
    },
    'bridges': {
        'distance_inversion': DistanceInversionEngine,
        'serendipity': SerendipityEngine,
        'cross_modal': CrossModalBridgeEngine,
    },
    'consolidation': {
        'intelligent': IntelligentConsolidationEngine,
        'incremental': IncrementalConsolidationEngine,
    }
}

def get_engine_class(engine_type: str, engine_name: str):
    """
    @TODO: Get engine class by type and name for dynamic loading.
    
    AGENTIC EMPOWERMENT: Dynamic engine loading enables
    flexible cognitive processing strategies and A/B testing.
    """
    return ENGINE_REGISTRY.get(engine_type, {}).get(engine_name)

def list_available_engines() -> dict:
    """
    @TODO: List all available cognitive engines.
    
    AGENTIC EMPOWERMENT: Engine discovery enables
    dynamic configuration and experimentation.
    """
    return {
        engine_type: list(engines.keys()) 
        for engine_type, engines in ENGINE_REGISTRY.items()
    }

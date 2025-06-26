# Phase 3: Advanced Features - Hyper-Detailed Implementation Tasks
*Week 3 of Development - Bridge Discovery & Advanced Dimensions*

## Overview
Phase 3 implements the advanced cognitive features that make the system truly intelligent:
- Bridge discovery with distance inversion algorithm
- Advanced dimension extractors (Social, Causal, Strategic)
- Consulting-specific enhancements
- Performance optimization with caching

## Prerequisites
- Phase 1 complete: SQLite, ONNX, Qdrant, basic extraction, 400D vectors
- Phase 2 complete: Activation spreading, project isolation, stakeholder filtering
- Docker running with Qdrant
- Test data available from Phase 2

## Day 1: Bridge Discovery Core Implementation

### Task 3.1.1: Create Bridge Discovery Engine Structure
**Time**: 30 minutes
**Priority**: 🔴 Critical

**Location**: `src/cognitive/bridges/engine.py`

**Implementation**:
```python
from dataclasses import dataclass
from typing import List, Set, Dict, Optional, Tuple
import numpy as np
from datetime import datetime
import asyncio
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class BridgeMemory:
    """Represents a discovered bridge between memory networks"""
    memory_id: str
    memory_content: str
    bridge_score: float  # Combined novelty + connection score
    novelty_score: float  # How different from query (0-1)
    connection_potential: float  # Max connection to activated set
    connection_path: List[str]  # Path through memories
    explanation: str  # Why this is a bridge
    project_context: Optional[str] = None
    stakeholder_relevance: Optional[List[str]] = None

@dataclass
class BridgeDiscoveryConfig:
    """Configuration for bridge discovery"""
    novelty_weight: float = 0.6
    connection_weight: float = 0.4
    bridge_threshold: float = 0.7
    max_bridges: int = 5
    min_novelty: float = 0.5  # Minimum distance from query
    max_novelty: float = 0.95  # Maximum distance (avoid noise)
    cache_ttl_seconds: int = 3600
    enable_explanations: bool = True

class BridgeDiscoveryEngine:
    """Discovers non-obvious connections using distance inversion"""
    
    def __init__(self, 
                 vector_store,
                 memory_repo,
                 connection_repo,
                 config: BridgeDiscoveryConfig = None):
        self.vector_store = vector_store
        self.memory_repo = memory_repo
        self.connection_repo = connection_repo
        self.config = config or BridgeDiscoveryConfig()
        self._cache = {}  # Simple in-memory cache
```

**Test Command**:
```bash
# Create test file
cat > tests/test_bridge_engine.py << 'EOF'
import pytest
from src.cognitive.bridges.engine import BridgeDiscoveryEngine, BridgeDiscoveryConfig

def test_bridge_engine_initialization():
    config = BridgeDiscoveryConfig(novelty_weight=0.7, connection_weight=0.3)
    engine = BridgeDiscoveryEngine(
        vector_store=None,
        memory_repo=None,
        connection_repo=None,
        config=config
    )
    assert engine.config.novelty_weight == 0.7
    assert engine.config.bridge_threshold == 0.7
EOF

pytest tests/test_bridge_engine.py -v
```

**Success Criteria**:
- [ ] Classes properly typed
- [ ] Config has sensible defaults
- [ ] Test passes

---

### Task 3.1.2: Implement Distance Inversion Algorithm
**Time**: 2 hours
**Priority**: 🔴 Critical

**Location**: `src/cognitive/bridges/engine.py` (add to class)

**Implementation**:
```python
async def discover_bridges(self, 
                         query_vector: np.ndarray,
                         activated_memories: Set[str],
                         context: Optional[Dict] = None) -> List[BridgeMemory]:
    """
    Discover bridge memories using distance inversion algorithm.
    
    The algorithm:
    1. Find memories distant from query (novel)
    2. But connected to activated memories
    3. Score by novelty * connection potential
    """
    # Check cache first
    cache_key = self._get_cache_key(query_vector, activated_memories)
    if cache_key in self._cache:
        cached_result, timestamp = self._cache[cache_key]
        if (datetime.now() - timestamp).seconds < self.config.cache_ttl_seconds:
            return cached_result
    
    # Search for distant memories
    all_memories = await self._search_novel_memories(query_vector, context)
    
    # Calculate connections to activated set
    bridges = await self._evaluate_bridge_potential(
        all_memories, 
        activated_memories,
        query_vector
    )
    
    # Generate explanations if enabled
    if self.config.enable_explanations:
        bridges = await self._generate_explanations(bridges, activated_memories)
    
    # Cache results
    self._cache[cache_key] = (bridges, datetime.now())
    
    return bridges

async def _search_novel_memories(self, 
                               query_vector: np.ndarray,
                               context: Optional[Dict]) -> List[Dict]:
    """Search for memories with controlled novelty"""
    
    # Search broadly first
    search_params = {
        "collection": "cognitive_episodes",
        "query_vector": query_vector,
        "limit": 1000,  # Cast wide net
    }
    
    # Add project filtering if context provided
    if context and context.get("project_id"):
        search_params["filter"] = {
            "must": [{"key": "project_id", "match": {"value": context["project_id"]}}]
        }
    
    results = await self.vector_store.search(**search_params)
    
    # Filter by novelty range
    novel_memories = []
    for result in results:
        similarity = result.score
        novelty = 1.0 - similarity
        
        if self.config.min_novelty <= novelty <= self.config.max_novelty:
            novel_memories.append({
                "id": result.payload["memory_id"],
                "score": similarity,
                "novelty": novelty,
                "payload": result.payload
            })
    
    return novel_memories

async def _evaluate_bridge_potential(self,
                                   candidates: List[Dict],
                                   activated_memories: Set[str],
                                   query_vector: np.ndarray) -> List[BridgeMemory]:
    """Evaluate bridge potential for each candidate"""
    
    # Pre-fetch activated memory vectors for efficiency
    activated_vectors = await self._fetch_memory_vectors(activated_memories)
    
    bridges = []
    
    for candidate in candidates:
        # Skip if already activated
        if candidate["id"] in activated_memories:
            continue
        
        # Get candidate vector
        memory = await self.memory_repo.get_by_id(candidate["id"])
        if not memory or not memory.embedding:
            continue
        
        # Calculate max connection to activated set
        max_connection = 0.0
        best_connection_id = None
        
        for activated_id, activated_vector in activated_vectors.items():
            similarity = cosine_similarity(
                memory.embedding.reshape(1, -1),
                activated_vector.reshape(1, -1)
            )[0][0]
            
            if similarity > max_connection:
                max_connection = similarity
                best_connection_id = activated_id
        
        # Calculate bridge score
        bridge_score = (
            self.config.novelty_weight * candidate["novelty"] +
            self.config.connection_weight * max_connection
        )
        
        if bridge_score >= self.config.bridge_threshold:
            bridges.append(BridgeMemory(
                memory_id=memory.id,
                memory_content=memory.content,
                bridge_score=bridge_score,
                novelty_score=candidate["novelty"],
                connection_potential=max_connection,
                connection_path=[best_connection_id, memory.id],
                explanation="",  # Will be filled if enabled
                project_context=memory.project_id,
                stakeholder_relevance=self._extract_stakeholders(memory.content)
            ))
    
    # Sort by bridge score and return top N
    bridges.sort(key=lambda b: b.bridge_score, reverse=True)
    return bridges[:self.config.max_bridges]
```

**Test Command**:
```bash
# Create integration test
cat > tests/test_bridge_discovery.py << 'EOF'
import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock

@pytest.mark.asyncio
async def test_distance_inversion():
    # Mock dependencies
    vector_store = AsyncMock()
    memory_repo = AsyncMock()
    
    # Create test data
    query_vector = np.random.rand(400)
    activated = {"mem1", "mem2"}
    
    # Mock search results - some novel, some not
    vector_store.search.return_value = [
        MagicMock(score=0.2, payload={"memory_id": "novel1"}),  # Very novel
        MagicMock(score=0.9, payload={"memory_id": "similar1"}),  # Too similar
        MagicMock(score=0.4, payload={"memory_id": "bridge1"}),  # Good bridge
    ]
    
    # Mock memory fetch
    memory_repo.get_by_id.return_value = MagicMock(
        id="bridge1",
        content="Strategic insight about competition",
        embedding=np.random.rand(400),
        project_id="proj1"
    )
    
    # Test
    from src.cognitive.bridges.engine import BridgeDiscoveryEngine
    engine = BridgeDiscoveryEngine(vector_store, memory_repo, None)
    
    bridges = await engine.discover_bridges(query_vector, activated)
    
    # Verify search was called
    vector_store.search.assert_called_once()
    
    # Verify filtering worked
    assert len(bridges) <= 5
EOF

pytest tests/test_bridge_discovery.py -v
```

**Performance Benchmark**:
```python
# Benchmark with realistic data
import time
query = np.random.rand(400)
activated = {f"mem{i}" for i in range(50)}

start = time.time()
bridges = await engine.discover_bridges(query, activated)
elapsed = time.time() - start

print(f"Bridge discovery took {elapsed:.3f}s for {len(bridges)} bridges")
assert elapsed < 1.0  # Must be under 1 second
```

**Success Criteria**:
- [ ] Distance inversion working (finds novel but connected)
- [ ] Respects novelty bounds (0.5-0.95)
- [ ] Returns max 5 bridges
- [ ] Performance < 1s
- [ ] Handles project context

---

### Task 3.1.3: Implement Bridge Explanation Generator
**Time**: 1 hour
**Priority**: 🟡 Medium

**Location**: `src/cognitive/bridges/engine.py` (add method)

**Implementation**:
```python
async def _generate_explanations(self, 
                               bridges: List[BridgeMemory],
                               activated_memories: Set[str]) -> List[BridgeMemory]:
    """Generate human-readable explanations for why each bridge is relevant"""
    
    # Fetch activated memory contents for context
    activated_contents = {}
    for mem_id in activated_memories:
        memory = await self.memory_repo.get_by_id(mem_id)
        if memory:
            activated_contents[mem_id] = memory.content
    
    for bridge in bridges:
        # Get the memory this bridge connects to
        connection_id = bridge.connection_path[0]
        connection_content = activated_contents.get(connection_id, "")
        
        # Analyze content similarity
        bridge_tokens = set(bridge.memory_content.lower().split())
        connection_tokens = set(connection_content.lower().split())
        shared_tokens = bridge_tokens & connection_tokens
        
        # Build explanation
        explanations = []
        
        # Explain novelty
        if bridge.novelty_score > 0.8:
            explanations.append("Highly novel perspective")
        elif bridge.novelty_score > 0.6:
            explanations.append("Different angle")
        
        # Explain connection
        if shared_tokens:
            shared_str = ", ".join(list(shared_tokens)[:3])
            explanations.append(f"Shares concepts: {shared_str}")
        
        # Explain potential value
        if bridge.connection_potential > 0.7:
            explanations.append("Strong conceptual link")
        
        # Add content type if available
        if hasattr(bridge, 'content_type'):
            explanations.append(f"Type: {bridge.content_type}")
        
        bridge.explanation = " | ".join(explanations)
    
    return bridges

def _get_cache_key(self, query_vector: np.ndarray, activated: Set[str]) -> str:
    """Generate cache key for results"""
    # Use first 10 dimensions of query + sorted activated IDs
    vector_key = "-".join(f"{x:.3f}" for x in query_vector[:10])
    activated_key = "-".join(sorted(list(activated)[:5]))  # First 5 IDs
    return f"{vector_key}:{activated_key}"

def _extract_stakeholders(self, content: str) -> List[str]:
    """Extract mentioned stakeholders from content"""
    # Simple pattern matching for MVP
    stakeholders = []
    
    # Look for @mentions
    import re
    mentions = re.findall(r'@(\w+)', content)
    stakeholders.extend(mentions)
    
    # Look for role indicators
    roles = ['CEO', 'CTO', 'VP', 'Director', 'Manager', 'Lead']
    for role in roles:
        if role.lower() in content.lower():
            stakeholders.append(role)
    
    return list(set(stakeholders))
```

**Test Command**:
```bash
# Test explanation generation
cat > tests/test_bridge_explanations.py << 'EOF'
import pytest
from src.cognitive.bridges.engine import BridgeMemory

@pytest.mark.asyncio
async def test_explanation_generation():
    bridge = BridgeMemory(
        memory_id="b1",
        memory_content="The CEO mentioned exploring new AI strategies for Q2",
        bridge_score=0.85,
        novelty_score=0.7,
        connection_potential=0.8,
        connection_path=["activated1", "b1"],
        explanation=""
    )
    
    # Mock activated memory
    activated_contents = {
        "activated1": "We need to improve our AI capabilities"
    }
    
    # After explanation generation
    assert "Different angle" in bridge.explanation
    assert "Strong conceptual link" in bridge.explanation
    assert "AI" in bridge.explanation or "capabilities" in bridge.explanation
EOF

pytest tests/test_bridge_explanations.py -v
```

**Success Criteria**:
- [ ] Explanations are meaningful
- [ ] Include novelty level
- [ ] Show shared concepts
- [ ] Extract stakeholders
- [ ] Cache key generation works

---

## Day 2: Advanced Dimension Extractors

### Task 3.2.1: Implement Social Dimension Extractor (Enhanced)
**Time**: 1.5 hours
**Priority**: 🔴 Critical

**Location**: `src/features/dimensions/social.py`

**Implementation**:
```python
import re
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class SocialContext:
    """Extracted social context from text"""
    mentioned_people: List[str]
    roles: List[str]
    departments: List[str]
    interaction_type: str  # directive, collaborative, informative
    authority_level: float  # 0-1
    audience_size: str  # individual, small_group, large_group, everyone

class SocialDimensionExtractor:
    """
    Extracts 3D social features for consulting context:
    1. Speaker Authority (0-1)
    2. Audience Relevance (0-1)  
    3. Interaction Score (0-1)
    """
    
    def __init__(self):
        # Authority indicators
        self.authority_markers = {
            'high': ['decide', 'approve', 'authorize', 'direct', 'mandate', 
                    'require', 'instruct', 'command', 'establish', 'determine'],
            'medium': ['recommend', 'suggest', 'propose', 'advise', 'consider',
                      'should', 'could', 'might', 'perhaps', 'possibly'],
            'low': ['think', 'feel', 'believe', 'wonder', 'guess',
                   'maybe', 'uncertain', 'unsure', 'questioning']
        }
        
        # Role patterns for consulting
        self.role_patterns = {
            'executive': r'\b(CEO|CFO|CTO|COO|President|VP|Vice President|Director)\b',
            'manager': r'\b(Manager|Lead|Head|Supervisor|Coordinator)\b',
            'consultant': r'\b(Consultant|Advisor|Analyst|Expert|Specialist)\b',
            'stakeholder': r'\b(Client|Customer|Partner|Vendor|Supplier)\b'
        }
        
        # Audience indicators
        self.audience_patterns = {
            'individual': r'\b(you|your|@\w+)\b',
            'small_group': r'\b(team|group|committee|board)\b',
            'large_group': r'\b(department|division|organization|company)\b',
            'everyone': r'\b(all|everyone|everybody|entire)\b'
        }
        
        # Interaction patterns
        self.interaction_patterns = {
            'directive': r'\b(must|will|shall|need to|have to|required)\b',
            'collaborative': r'\b(let\'s|we should|together|collaborate|work with)\b',
            'questioning': r'\b(what|why|how|when|where|who|\?)\b',
            'informative': r'\b(is|are|was|were|has been|have been)\b'
        }
    
    def extract(self, text: str, speaker: Optional[str] = None) -> np.ndarray:
        """Extract 3D social features from text"""
        
        # Extract social context
        context = self._extract_social_context(text)
        
        # 1. Speaker Authority (0-1)
        authority = self._calculate_authority(text, context, speaker)
        
        # 2. Audience Relevance (0-1)
        relevance = self._calculate_audience_relevance(text, context)
        
        # 3. Interaction Score (0-1)
        interaction = self._calculate_interaction_score(text, context)
        
        return np.array([authority, relevance, interaction])
    
    def _extract_social_context(self, text: str) -> SocialContext:
        """Extract social elements from text"""
        
        # Find mentioned people
        mentioned_people = re.findall(r'@(\w+)', text)
        mentioned_people.extend(re.findall(r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b', text))
        
        # Find roles
        roles = []
        for role_type, pattern in self.role_patterns.items():
            if re.search(pattern, text, re.I):
                roles.append(role_type)
        
        # Find departments
        dept_pattern = r'\b(Sales|Marketing|Engineering|HR|Finance|IT|Legal|Operations)\b'
        departments = re.findall(dept_pattern, text, re.I)
        
        # Determine interaction type
        interaction_type = self._determine_interaction_type(text)
        
        # Calculate authority level
        authority_level = self._calculate_raw_authority(text)
        
        # Determine audience size
        audience_size = self._determine_audience_size(text)
        
        return SocialContext(
            mentioned_people=list(set(mentioned_people)),
            roles=roles,
            departments=list(set(departments)),
            interaction_type=interaction_type,
            authority_level=authority_level,
            audience_size=audience_size
        )
    
    def _calculate_authority(self, text: str, context: SocialContext, 
                           speaker: Optional[str]) -> float:
        """Calculate speaker authority score"""
        
        score = 0.0
        
        # Check authority markers
        high_count = sum(1 for marker in self.authority_markers['high'] 
                        if marker in text.lower())
        medium_count = sum(1 for marker in self.authority_markers['medium'] 
                          if marker in text.lower())
        low_count = sum(1 for marker in self.authority_markers['low'] 
                       if marker in text.lower())
        
        # Weight authority markers
        if high_count > 0:
            score += min(high_count * 0.3, 0.6)
        if medium_count > 0:
            score += min(medium_count * 0.1, 0.2)
        if low_count > 0:
            score -= min(low_count * 0.1, 0.2)
        
        # Boost for executive roles
        if 'executive' in context.roles:
            score += 0.3
        elif 'manager' in context.roles:
            score += 0.2
        
        # Boost for directive interaction
        if context.interaction_type == 'directive':
            score += 0.1
        
        return np.clip(score, 0, 1)
    
    def _calculate_audience_relevance(self, text: str, context: SocialContext) -> float:
        """Calculate audience relevance score"""
        
        score = 0.0
        
        # Direct mentions are highly relevant
        if context.mentioned_people:
            score += 0.4
        
        # Department mentions
        if context.departments:
            score += 0.2
        
        # Audience size impacts relevance
        audience_scores = {
            'individual': 0.4,
            'small_group': 0.3,
            'large_group': 0.2,
            'everyone': 0.3
        }
        score += audience_scores.get(context.audience_size, 0.1)
        
        return np.clip(score, 0, 1)
    
    def _calculate_interaction_score(self, text: str, context: SocialContext) -> float:
        """Calculate interaction dynamics score"""
        
        score = 0.0
        
        # Count interaction patterns
        patterns_found = 0
        for pattern_type, pattern in self.interaction_patterns.items():
            if re.search(pattern, text, re.I):
                patterns_found += 1
                
                # Weight different interaction types
                if pattern_type == 'collaborative':
                    score += 0.3
                elif pattern_type == 'directive':
                    score += 0.2
                elif pattern_type == 'questioning':
                    score += 0.25
                else:
                    score += 0.1
        
        # Boost for multiple people mentioned
        if len(context.mentioned_people) > 1:
            score += 0.2
        
        # Boost for cross-department interaction
        if len(context.departments) > 1:
            score += 0.15
        
        return np.clip(score, 0, 1)
    
    def _determine_interaction_type(self, text: str) -> str:
        """Determine primary interaction type"""
        
        # Count each type
        counts = {}
        for itype, pattern in self.interaction_patterns.items():
            counts[itype] = len(re.findall(pattern, text, re.I))
        
        # Return type with most matches
        if counts:
            return max(counts, key=counts.get)
        return 'informative'
    
    def _calculate_raw_authority(self, text: str) -> float:
        """Calculate raw authority level from text"""
        
        # Count authority indicators
        high = sum(1 for m in self.authority_markers['high'] if m in text.lower())
        low = sum(1 for m in self.authority_markers['low'] if m in text.lower())
        
        # Simple scoring
        score = (high - low) / 10.0
        return np.clip(score + 0.5, 0, 1)
    
    def _determine_audience_size(self, text: str) -> str:
        """Determine audience size from text"""
        
        for size, pattern in self.audience_patterns.items():
            if re.search(pattern, text, re.I):
                return size
        return 'small_group'  # Default
```

**Test Command**:
```bash
# Create comprehensive test
cat > tests/test_social_dimension.py << 'EOF'
import pytest
import numpy as np
from src.features.dimensions.social import SocialDimensionExtractor

def test_social_extraction():
    extractor = SocialDimensionExtractor()
    
    # Test authority detection
    text1 = "As CEO, I've decided we must pivot our strategy immediately"
    features1 = extractor.extract(text1, speaker="John Smith")
    assert features1[0] > 0.7  # High authority
    
    # Test audience relevance
    text2 = "@sarah @mike The marketing team needs to review this"
    features2 = extractor.extract(text2)
    assert features2[1] > 0.6  # High relevance (mentions + department)
    
    # Test interaction score
    text3 = "Let's collaborate with the sales team to improve our approach"
    features3 = extractor.extract(text3)
    assert features3[2] > 0.5  # Good interaction (collaborative + department)
    
    # Test all dimensions are in range
    assert all(0 <= f <= 1 for f in features1)
    assert all(0 <= f <= 1 for f in features2)
    assert all(0 <= f <= 1 for f in features3)

def test_edge_cases():
    extractor = SocialDimensionExtractor()
    
    # Empty text
    features = extractor.extract("")
    assert features.shape == (3,)
    assert all(f >= 0 for f in features)
    
    # Very long text
    long_text = " ".join(["meeting discussion"] * 1000)
    features = extractor.extract(long_text)
    assert features.shape == (3,)
EOF

pytest tests/test_social_dimension.py -v
```

**Success Criteria**:
- [ ] Detects authority markers correctly
- [ ] Identifies audience and relevance
- [ ] Measures interaction dynamics
- [ ] Returns normalized 3D vector
- [ ] Handles consulting context (roles, departments)

---

### Task 3.2.2: Implement Causal Dimension Extractor
**Time**: 1.5 hours
**Priority**: 🔴 Critical

**Location**: `src/features/dimensions/causal.py`

**Implementation**:
```python
import re
import numpy as np
from typing import List, Dict, Tuple, Optional
from enum import Enum

class CausalType(Enum):
    CAUSE = "cause"
    EFFECT = "effect"
    CORRELATION = "correlation"
    PREVENTION = "prevention"
    ENABLEMENT = "enablement"

class CausalDimensionExtractor:
    """
    Extracts 3D causal features:
    1. Cause-Effect Strength (0-1)
    2. Logical Coherence (0-1)
    3. Impact Scope (0-1)
    """
    
    def __init__(self):
        # Causal indicators
        self.causal_patterns = {
            'cause': [
                r'\b(because|since|as|due to|caused by|resulted from|stems from)\b',
                r'\b(leads? to|results? in|causes?|creates?|produces?|generates?)\b',
                r'\b(therefore|thus|hence|consequently|as a result|so)\b'
            ],
            'effect': [
                r'\b(resulted in|led to|caused|created|produced|generated)\b',
                r'\b(impact|outcome|consequence|result|effect|implication)\b'
            ],
            'correlation': [
                r'\b(correlates? with|associated with|linked to|related to)\b',
                r'\b(connection between|relationship between)\b'
            ],
            'prevention': [
                r'\b(prevents?|avoids?|stops?|blocks?|inhibits?|reduces?)\b',
                r'\b(mitigates?|minimizes?|eliminates?)\b'
            ],
            'enablement': [
                r'\b(enables?|allows?|permits?|facilitates?|supports?)\b',
                r'\b(makes? possible|makes? it possible)\b'
            ]
        }
        
        # Logical connectors
        self.logical_connectors = {
            'strong': ['therefore', 'thus', 'consequently', 'as a result', 'hence'],
            'medium': ['so', 'then', 'accordingly', 'for this reason'],
            'weak': ['maybe', 'possibly', 'might', 'could', 'perhaps']
        }
        
        # Impact indicators
        self.impact_patterns = {
            'high': r'\b(critical|crucial|essential|vital|fundamental|significant|major)\b',
            'medium': r'\b(important|notable|considerable|meaningful|relevant)\b',
            'low': r'\b(minor|slight|small|marginal|minimal|negligible)\b'
        }
        
        # Scope indicators
        self.scope_patterns = {
            'global': r'\b(entire|whole|all|company-wide|organization-wide|across)\b',
            'department': r'\b(department|division|team|group|unit)\b',
            'individual': r'\b(personal|individual|specific|particular)\b'
        }
    
    def extract(self, text: str) -> np.ndarray:
        """Extract 3D causal features from text"""
        
        # 1. Cause-Effect Strength
        cause_effect = self._calculate_cause_effect_strength(text)
        
        # 2. Logical Coherence
        coherence = self._calculate_logical_coherence(text)
        
        # 3. Impact Scope
        impact = self._calculate_impact_scope(text)
        
        return np.array([cause_effect, coherence, impact])
    
    def _calculate_cause_effect_strength(self, text: str) -> float:
        """Calculate strength of causal relationships"""
        
        score = 0.0
        causal_count = 0
        
        # Check for causal patterns
        for causal_type, patterns in self.causal_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.I)
                if matches:
                    causal_count += len(matches)
                    
                    # Weight different types
                    if causal_type == 'cause':
                        score += len(matches) * 0.3
                    elif causal_type == 'effect':
                        score += len(matches) * 0.25
                    elif causal_type == 'enablement':
                        score += len(matches) * 0.2
                    elif causal_type == 'prevention':
                        score += len(matches) * 0.15
                    else:  # correlation
                        score += len(matches) * 0.1
        
        # Normalize by text length (words)
        word_count = len(text.split())
        if word_count > 0:
            score = score / np.sqrt(word_count) * 10
        
        return np.clip(score, 0, 1)
    
    def _calculate_logical_coherence(self, text: str) -> float:
        """Calculate logical coherence of reasoning"""
        
        score = 0.0
        
        # Check logical connectors
        strong_connectors = sum(1 for conn in self.logical_connectors['strong'] 
                               if conn.lower() in text.lower())
        medium_connectors = sum(1 for conn in self.logical_connectors['medium'] 
                               if conn.lower() in text.lower())
        weak_connectors = sum(1 for conn in self.logical_connectors['weak'] 
                             if conn.lower() in text.lower())
        
        # Weight connectors
        score += strong_connectors * 0.4
        score += medium_connectors * 0.2
        score -= weak_connectors * 0.1  # Weak connectors reduce coherence
        
        # Check for structured reasoning (if-then patterns)
        if_then_pattern = r'\b(if\s.+then\s|when\s.+then\s)'
        if_then_matches = len(re.findall(if_then_pattern, text, re.I))
        score += if_then_matches * 0.3
        
        # Check for evidence markers
        evidence_pattern = r'\b(data shows?|research indicates?|studies? show|evidence suggests?)\b'
        evidence_matches = len(re.findall(evidence_pattern, text, re.I))
        score += evidence_matches * 0.2
        
        return np.clip(score, 0, 1)
    
    def _calculate_impact_scope(self, text: str) -> float:
        """Calculate scope and magnitude of impact"""
        
        # Detect impact level
        high_impact = len(re.findall(self.impact_patterns['high'], text, re.I))
        medium_impact = len(re.findall(self.impact_patterns['medium'], text, re.I))
        low_impact = len(re.findall(self.impact_patterns['low'], text, re.I))
        
        # Calculate impact score
        impact_score = (high_impact * 0.5 + medium_impact * 0.3 - low_impact * 0.1)
        
        # Detect scope
        global_scope = len(re.findall(self.scope_patterns['global'], text, re.I))
        dept_scope = len(re.findall(self.scope_patterns['department'], text, re.I))
        indiv_scope = len(re.findall(self.scope_patterns['individual'], text, re.I))
        
        # Calculate scope score
        scope_score = (global_scope * 0.5 + dept_scope * 0.3 + indiv_scope * 0.1)
        
        # Combine impact and scope
        combined_score = (impact_score + scope_score) / 2
        
        # Boost for quantified impact
        quantified_pattern = r'\b(\d+%|\$\d+|\d+\s*(million|billion|thousand))\b'
        if re.search(quantified_pattern, text):
            combined_score += 0.2
        
        return np.clip(combined_score, 0, 1)
    
    def identify_causal_chains(self, text: str) -> List[Tuple[str, str, float]]:
        """
        Identify causal chains in text.
        Returns list of (cause, effect, confidence) tuples.
        """
        chains = []
        
        # Pattern for "X causes/leads to Y"
        direct_pattern = r'(\b[^.]+?)\s+(?:causes?|leads? to|results? in)\s+([^.]+)'
        matches = re.findall(direct_pattern, text, re.I)
        
        for cause, effect in matches:
            # Clean up matches
            cause = cause.strip().lower()
            effect = effect.strip().lower()
            
            # Skip if too short
            if len(cause.split()) < 2 or len(effect.split()) < 2:
                continue
            
            # Calculate confidence based on certainty words
            confidence = 0.7  # Base confidence
            if any(word in effect for word in ['will', 'must', 'always']):
                confidence = 0.9
            elif any(word in effect for word in ['might', 'could', 'may']):
                confidence = 0.5
            
            chains.append((cause, effect, confidence))
        
        return chains[:3]  # Return top 3 chains
```

**Test Command**:
```bash
# Create causal dimension tests
cat > tests/test_causal_dimension.py << 'EOF'
import pytest
import numpy as np
from src.features.dimensions.causal import CausalDimensionExtractor

def test_causal_extraction():
    extractor = CausalDimensionExtractor()
    
    # Test strong causation
    text1 = "Because we reduced costs, therefore our profits increased significantly"
    features1 = extractor.extract(text1)
    assert features1[0] > 0.6  # Strong cause-effect
    assert features1[1] > 0.5  # Good coherence (therefore)
    
    # Test impact scope
    text2 = "This critical decision will impact the entire organization"
    features2 = extractor.extract(text2)
    assert features2[2] > 0.7  # High impact + global scope
    
    # Test weak causation
    text3 = "This might possibly lead to some minor improvements"
    features3 = extractor.extract(text3)
    assert features3[0] < 0.5  # Weak causation
    assert features3[1] < 0.5  # Low coherence (weak connectors)
    
    # Test causal chains
    text4 = "Poor communication causes delays which results in customer dissatisfaction"
    chains = extractor.identify_causal_chains(text4)
    assert len(chains) > 0
    assert chains[0][2] > 0.5  # Reasonable confidence

def test_quantified_impact():
    extractor = CausalDimensionExtractor()
    
    text = "This will increase revenue by 45% across all departments"
    features = extractor.extract(text)
    assert features[2] > 0.6  # Quantified impact boosts score
EOF

pytest tests/test_causal_dimension.py -v
```

**Success Criteria**:
- [ ] Detects causal relationships
- [ ] Measures logical coherence
- [ ] Calculates impact scope
- [ ] Identifies causal chains
- [ ] Handles quantified impacts

---

### Task 3.2.3: Implement Strategic Dimension Extractor (Replacing Evolutionary)
**Time**: 2 hours
**Priority**: 🔴 Critical

**Location**: `src/features/dimensions/strategic.py`

**Implementation**:
```python
import re
import numpy as np
from typing import Dict, List, Set, Optional
from datetime import datetime
from enum import Enum

class StrategyType(Enum):
    GROWTH = "growth"
    EFFICIENCY = "efficiency"
    INNOVATION = "innovation"
    RISK_MITIGATION = "risk_mitigation"
    COMPETITIVE = "competitive"
    TRANSFORMATION = "transformation"

class StrategicDimensionExtractor:
    """
    Extracts 3D strategic features for consulting context:
    1. Strategic Alignment (0-1) - How well aligned with strategy
    2. Time Horizon (0-1) - Short-term (0) to Long-term (1)
    3. Risk/Opportunity Balance (0-1) - Risk-focused (0) to Opportunity-focused (1)
    """
    
    def __init__(self):
        # Strategic indicators
        self.strategy_patterns = {
            'growth': [
                r'\b(grow|growth|expand|expansion|scale|scaling|increase)\b',
                r'\b(market share|revenue growth|customer acquisition)\b',
                r'\b(new markets?|new products?|new services?)\b'
            ],
            'efficiency': [
                r'\b(optimize|optimization|efficient|efficiency|streamline)\b',
                r'\b(cost reduction|reduce costs?|save money|savings)\b',
                r'\b(automate|automation|process improvement)\b'
            ],
            'innovation': [
                r'\b(innovate|innovation|innovative|disrupt|disruption)\b',
                r'\b(new technology|emerging tech|digital transformation)\b',
                r'\b(research|R&D|experiment|pilot)\b'
            ],
            'risk_mitigation': [
                r'\b(risk|risks|mitigate|mitigation|hedge|protect)\b',
                r'\b(compliance|regulatory|security|safety)\b',
                r'\b(contingency|backup|failsafe|redundancy)\b'
            ],
            'competitive': [
                r'\b(competitive|competition|competitor|differentiate)\b',
                r'\b(market position|competitive advantage|outperform)\b',
                r'\b(benchmark|best-in-class|industry leader)\b'
            ],
            'transformation': [
                r'\b(transform|transformation|change management|restructure)\b',
                r'\b(cultural change|organizational change|pivot)\b',
                r'\b(modernize|modernization|reinvent)\b'
            ]
        }
        
        # Time horizon indicators
        self.time_patterns = {
            'immediate': [r'\b(now|today|immediately|urgent|asap)\b', 0.0],
            'short_term': [r'\b(this week|next week|this month|Q1|Q2)\b', 0.2],
            'medium_term': [r'\b(this year|next year|annual|yearly)\b', 0.5],
            'long_term': [r'\b(3-5 years?|5 years?|long-term|strategic|future)\b', 0.8],
            'visionary': [r'\b(10 years?|decade|vision|2030|2035)\b', 1.0]
        }
        
        # Risk vs Opportunity indicators
        self.risk_opportunity_patterns = {
            'risk': [
                r'\b(risk|threat|danger|vulnerability|exposure)\b',
                r'\b(downside|negative|adverse|unfavorable)\b',
                r'\b(avoid|prevent|protect|defend|mitigate)\b'
            ],
            'opportunity': [
                r'\b(opportunity|potential|possibility|upside)\b',
                r'\b(leverage|capitalize|exploit|benefit|advantage)\b',
                r'\b(growth|expand|develop|create|build)\b'
            ]
        }
        
        # Strategic value indicators
        self.value_patterns = {
            'high_value': r'\b(strategic|critical|key|core|essential|vital)\b',
            'medium_value': r'\b(important|significant|valuable|relevant)\b',
            'low_value': r'\b(tactical|operational|minor|routine)\b'
        }
        
        # Consulting-specific patterns
        self.consulting_patterns = {
            'recommendation': r'\b(recommend|advise|suggest|propose)\b',
            'analysis': r'\b(analyze|analysis|assess|evaluate|review)\b',
            'hypothesis': r'\b(hypothesis|hypothesize|assume|believe)\b',
            'evidence': r'\b(data shows?|evidence|proof|validate|confirm)\b'
        }
    
    def extract(self, text: str) -> np.ndarray:
        """Extract 3D strategic features from text"""
        
        # 1. Strategic Alignment
        alignment = self._calculate_strategic_alignment(text)
        
        # 2. Time Horizon
        horizon = self._calculate_time_horizon(text)
        
        # 3. Risk/Opportunity Balance
        balance = self._calculate_risk_opportunity_balance(text)
        
        return np.array([alignment, horizon, balance])
    
    def _calculate_strategic_alignment(self, text: str) -> float:
        """Calculate how strategically aligned the content is"""
        
        score = 0.0
        strategy_count = 0
        
        # Check each strategy type
        strategy_scores = {}
        for strategy_type, patterns in self.strategy_patterns.items():
            type_score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text, re.I))
                type_score += matches
            strategy_scores[strategy_type] = type_score
            strategy_count += type_score
        
        # Base alignment on presence of strategic thinking
        if strategy_count > 0:
            score += min(strategy_count * 0.15, 0.5)
        
        # Check strategic value
        high_value = len(re.findall(self.value_patterns['high_value'], text, re.I))
        medium_value = len(re.findall(self.value_patterns['medium_value'], text, re.I))
        low_value = len(re.findall(self.value_patterns['low_value'], text, re.I))
        
        score += high_value * 0.2
        score += medium_value * 0.1
        score -= low_value * 0.1  # Tactical focus reduces strategic alignment
        
        # Boost for consulting patterns (strategic thinking)
        for pattern in self.consulting_patterns.values():
            if re.search(pattern, text, re.I):
                score += 0.1
        
        # Boost for multiple strategy types (holistic thinking)
        active_strategies = sum(1 for s in strategy_scores.values() if s > 0)
        if active_strategies > 2:
            score += 0.2
        
        return np.clip(score, 0, 1)
    
    def _calculate_time_horizon(self, text: str) -> float:
        """Calculate time horizon (0=short-term, 1=long-term)"""
        
        horizon_scores = []
        
        # Check for time indicators
        for horizon_type, (pattern, score) in self.time_patterns.items():
            if re.search(pattern, text, re.I):
                horizon_scores.append(score)
        
        # If multiple horizons, take the longest
        if horizon_scores:
            base_score = max(horizon_scores)
        else:
            base_score = 0.3  # Default to short-medium term
        
        # Adjust based on strategic language
        if re.search(r'\b(vision|strategic plan|roadmap|future state)\b', text, re.I):
            base_score = max(base_score, 0.7)
        
        # Adjust based on planning language
        if re.search(r'\b(plan|planning|forecast|projection)\b', text, re.I):
            base_score += 0.1
        
        return np.clip(base_score, 0, 1)
    
    def _calculate_risk_opportunity_balance(self, text: str) -> float:
        """
        Calculate risk/opportunity balance
        0 = Risk-focused, 1 = Opportunity-focused, 0.5 = Balanced
        """
        
        # Count risk and opportunity mentions
        risk_count = 0
        for pattern in self.risk_opportunity_patterns['risk']:
            risk_count += len(re.findall(pattern, text, re.I))
        
        opportunity_count = 0
        for pattern in self.risk_opportunity_patterns['opportunity']:
            opportunity_count += len(re.findall(pattern, text, re.I))
        
        # Calculate balance
        total = risk_count + opportunity_count
        if total == 0:
            return 0.5  # Neutral if neither mentioned
        
        # Calculate opportunity ratio
        opportunity_ratio = opportunity_count / total
        
        # Adjust for mitigation language (balanced approach)
        if re.search(r'\b(balance|balanced|both|consider|weigh)\b', text, re.I):
            # Pull toward center
            opportunity_ratio = 0.5 + (opportunity_ratio - 0.5) * 0.7
        
        return opportunity_ratio
    
    def identify_strategic_initiatives(self, text: str) -> List[Dict[str, any]]:
        """
        Identify specific strategic initiatives mentioned.
        Returns list of initiatives with their type and confidence.
        """
        initiatives = []
        
        # Pattern for initiative detection
        initiative_patterns = [
            r'(?:we should|we will|we need to|plan to|going to)\s+([^.]+)',
            r'(?:initiative|project|program)\s+(?:to|for)\s+([^.]+)',
            r'(?:strategic|key|major)\s+(?:initiative|priority|focus):\s*([^.]+)'
        ]
        
        for pattern in initiative_patterns:
            matches = re.findall(pattern, text, re.I)
            for match in matches:
                # Clean and validate
                initiative_text = match.strip()
                if len(initiative_text.split()) < 3 or len(initiative_text) > 200:
                    continue
                
                # Determine type
                strategy_type = self._determine_strategy_type(initiative_text)
                
                # Calculate confidence
                confidence = 0.7
                if any(word in initiative_text.lower() for word in ['will', 'must', 'critical']):
                    confidence = 0.9
                elif any(word in initiative_text.lower() for word in ['might', 'could', 'consider']):
                    confidence = 0.5
                
                initiatives.append({
                    'text': initiative_text,
                    'type': strategy_type,
                    'confidence': confidence
                })
        
        return initiatives[:5]  # Return top 5
    
    def _determine_strategy_type(self, text: str) -> str:
        """Determine the primary strategy type for text"""
        
        scores = {}
        for strategy_type, patterns in self.strategy_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, text, re.I):
                    score += 1
            scores[strategy_type] = score
        
        if scores:
            return max(scores, key=scores.get)
        return 'general'
    
    def calculate_strategic_importance(self, 
                                     features: np.ndarray,
                                     context: Optional[Dict] = None) -> float:
        """
        Calculate overall strategic importance score.
        Used for prioritizing memories.
        """
        alignment, horizon, balance = features
        
        # Base importance on alignment
        importance = alignment * 0.5
        
        # Long-term thinking is valuable
        importance += horizon * 0.3
        
        # Balanced risk/opportunity is optimal
        balance_score = 1.0 - abs(balance - 0.5) * 2  # Peak at 0.5
        importance += balance_score * 0.2
        
        # Context adjustments
        if context:
            # Boost for executive meetings
            if context.get('meeting_type') == 'executive':
                importance *= 1.2
            
            # Boost for strategy sessions
            if 'strategy' in context.get('meeting_title', '').lower():
                importance *= 1.3
        
        return np.clip(importance, 0, 1)
```

**Test Command**:
```bash
# Create strategic dimension tests
cat > tests/test_strategic_dimension.py << 'EOF'
import pytest
import numpy as np
from src.features.dimensions.strategic import StrategicDimensionExtractor

def test_strategic_extraction():
    extractor = StrategicDimensionExtractor()
    
    # Test high strategic alignment
    text1 = "Our strategic initiative to expand into new markets will drive long-term growth"
    features1 = extractor.extract(text1)
    assert features1[0] > 0.6  # High alignment (strategic + growth)
    assert features1[1] > 0.7  # Long-term horizon
    assert features1[2] > 0.5  # Opportunity-focused
    
    # Test risk-focused content
    text2 = "We must mitigate risks and protect our current position from threats"
    features2 = extractor.extract(text2)
    assert features2[2] < 0.4  # Risk-focused
    
    # Test balanced approach
    text3 = "We should balance growth opportunities with risk mitigation"
    features3 = extractor.extract(text3)
    assert 0.4 < features3[2] < 0.6  # Balanced
    
    # Test initiative identification
    text4 = "We will launch a digital transformation initiative to modernize operations"
    initiatives = extractor.identify_strategic_initiatives(text4)
    assert len(initiatives) > 0
    assert initiatives[0]['type'] in ['transformation', 'innovation']
    
def test_strategic_importance():
    extractor = StrategicDimensionExtractor()
    
    # High importance: aligned, long-term, balanced
    features = np.array([0.8, 0.8, 0.5])
    importance = extractor.calculate_strategic_importance(features)
    assert importance > 0.7
    
    # Low importance: tactical, short-term, extreme
    features = np.array([0.2, 0.1, 0.9])
    importance = extractor.calculate_strategic_importance(features)
    assert importance < 0.4

def test_time_horizon_detection():
    extractor = StrategicDimensionExtractor()
    
    # Immediate
    text1 = "We need to address this immediately"
    assert extractor.extract(text1)[1] < 0.3
    
    # Long-term
    text2 = "Our 5-year strategic vision focuses on transformation"
    assert extractor.extract(text2)[1] > 0.7
EOF

pytest tests/test_strategic_dimension.py -v
```

**Success Criteria**:
- [ ] Detects strategic alignment
- [ ] Measures time horizon correctly
- [ ] Calculates risk/opportunity balance
- [ ] Identifies strategic initiatives
- [ ] Calculates strategic importance
- [ ] Handles consulting context

---

## Day 3: Integration and Optimization

### Task 3.3.1: Integrate Bridge Discovery with Activation Engine
**Time**: 1.5 hours
**Priority**: 🔴 Critical

**Location**: `src/api/cognitive_routes.py`

**Implementation**:
```python
from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List, Dict
import time
import numpy as np
from pydantic import BaseModel, Field

from src.cognitive.activation.engine import ActivationEngine
from src.cognitive.bridges.engine import BridgeDiscoveryEngine
from src.models.api_models import CognitiveQueryRequest, CognitiveQueryResponse

router = APIRouter(prefix="/api/v1/cognitive", tags=["cognitive"])

class BridgeResult(BaseModel):
    """Bridge discovery result for API"""
    memory_id: str
    content: str
    bridge_score: float = Field(..., ge=0, le=1)
    novelty_score: float = Field(..., ge=0, le=1)
    connection_strength: float = Field(..., ge=0, le=1)
    explanation: str
    connection_path: List[str]
    project_context: Optional[str] = None

class EnhancedCognitiveResponse(BaseModel):
    """Enhanced response with bridges"""
    activated_memories: List[Dict]
    bridge_discoveries: List[BridgeResult]
    activation_stats: Dict[str, int]
    processing_time_ms: float
    query_understanding: Dict[str, float]

@router.post("/query/enhanced", response_model=EnhancedCognitiveResponse)
async def enhanced_cognitive_query(request: CognitiveQueryRequest):
    """
    Perform cognitive query with activation spreading and bridge discovery.
    Integrates both algorithms for comprehensive results.
    """
    start_time = time.time()
    
    try:
        # 1. Generate query embedding with enhanced dimensions
        query_vector = await generate_enhanced_query_vector(request.query)
        
        # 2. Run activation spreading
        activation_result = await activation_engine.spread_activation(
            query_vector=query_vector,
            context={
                "project_id": request.project_id,
                "stakeholder_filter": request.stakeholder_filter,
                "meeting_type_filter": request.meeting_type_filter
            }
        )
        
        # 3. Get activated memory IDs for bridge discovery
        activated_ids = {
            mem.memory_id 
            for mem in activation_result.all_memories()
        }
        
        # 4. Discover bridges if requested
        bridges = []
        if request.include_bridges and activated_ids:
            bridge_memories = await bridge_engine.discover_bridges(
                query_vector=query_vector,
                activated_memories=activated_ids,
                context={"project_id": request.project_id}
            )
            
            # Convert to API format
            for bridge in bridge_memories:
                bridges.append(BridgeResult(
                    memory_id=bridge.memory_id,
                    content=bridge.memory_content,
                    bridge_score=bridge.bridge_score,
                    novelty_score=bridge.novelty_score,
                    connection_strength=bridge.connection_potential,
                    explanation=bridge.explanation,
                    connection_path=bridge.connection_path,
                    project_context=bridge.project_context
                ))
        
        # 5. Calculate statistics
        stats = {
            "core_memories": len(activation_result.core_memories),
            "contextual_memories": len(activation_result.contextual_memories),
            "peripheral_memories": len(activation_result.peripheral_memories),
            "bridges_found": len(bridges),
            "total_activated": len(activated_ids)
        }
        
        # 6. Analyze query understanding
        query_understanding = await analyze_query_dimensions(query_vector)
        
        processing_time = (time.time() - start_time) * 1000
        
        return EnhancedCognitiveResponse(
            activated_memories=activation_result.to_dict()["memories"],
            bridge_discoveries=bridges,
            activation_stats=stats,
            processing_time_ms=processing_time,
            query_understanding=query_understanding
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def generate_enhanced_query_vector(query: str) -> np.ndarray:
    """Generate 400D query vector with all dimension extractors"""
    
    # Get base embedding (384D)
    semantic_embedding = await onnx_encoder.encode(query)
    
    # Extract all dimensions (16D)
    temporal = temporal_extractor.extract(query)  # 4D
    emotional = emotional_extractor.extract(query)  # 3D
    social = social_extractor.extract(query)  # 3D
    causal = causal_extractor.extract(query)  # 3D
    strategic = strategic_extractor.extract(query)  # 3D
    
    # Combine into 400D vector
    dimensions = np.concatenate([
        temporal, emotional, social, causal, strategic
    ])
    
    return np.concatenate([semantic_embedding, dimensions])

async def analyze_query_dimensions(query_vector: np.ndarray) -> Dict[str, float]:
    """Analyze the query vector to understand query intent"""
    
    # Extract dimension values
    dimensions = query_vector[-16:]  # Last 16 dimensions
    
    # Map to named dimensions
    return {
        "temporal_urgency": float(dimensions[0]),
        "temporal_deadline": float(dimensions[1]),
        "emotional_sentiment": float(dimensions[4]),
        "social_authority": float(dimensions[7]),
        "social_relevance": float(dimensions[8]),
        "causal_strength": float(dimensions[10]),
        "strategic_alignment": float(dimensions[13]),
        "strategic_horizon": float(dimensions[14])
    }

@router.get("/bridges/explain/{memory_id}")
async def explain_bridge(
    memory_id: str,
    query: str = Query(..., description="Original query"),
    activated_id: str = Query(..., description="ID of activated memory it connects to")
):
    """
    Get detailed explanation of why a memory is a good bridge.
    Useful for understanding the algorithm's reasoning.
    """
    try:
        # Get both memories
        bridge_memory = await memory_repo.get_by_id(memory_id)
        activated_memory = await memory_repo.get_by_id(activated_id)
        
        if not bridge_memory or not activated_memory:
            raise HTTPException(status_code=404, detail="Memory not found")
        
        # Generate query vector
        query_vector = await generate_enhanced_query_vector(query)
        
        # Calculate similarities
        query_similarity = cosine_similarity(
            query_vector.reshape(1, -1),
            bridge_memory.embedding.reshape(1, -1)
        )[0][0]
        
        connection_similarity = cosine_similarity(
            bridge_memory.embedding.reshape(1, -1),
            activated_memory.embedding.reshape(1, -1)
        )[0][0]
        
        # Analyze shared concepts
        bridge_tokens = set(bridge_memory.content.lower().split())
        activated_tokens = set(activated_memory.content.lower().split())
        query_tokens = set(query.lower().split())
        
        shared_with_activated = bridge_tokens & activated_tokens
        shared_with_query = bridge_tokens & query_tokens
        unique_concepts = bridge_tokens - activated_tokens - query_tokens
        
        explanation = {
            "bridge_memory": {
                "id": memory_id,
                "content": bridge_memory.content,
                "type": bridge_memory.content_type
            },
            "connected_to": {
                "id": activated_id,
                "content": activated_memory.content
            },
            "metrics": {
                "novelty_from_query": 1.0 - query_similarity,
                "connection_strength": connection_similarity,
                "bridge_score": (0.6 * (1.0 - query_similarity) + 
                               0.4 * connection_similarity)
            },
            "concept_analysis": {
                "shared_with_activated": list(shared_with_activated)[:10],
                "shared_with_query": list(shared_with_query)[:10],
                "unique_concepts": list(unique_concepts)[:10]
            },
            "explanation": generate_bridge_explanation(
                novelty=1.0 - query_similarity,
                connection=connection_similarity,
                shared_concepts=shared_with_activated
            )
        }
        
        return explanation
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def generate_bridge_explanation(novelty: float, connection: float, 
                               shared_concepts: set) -> str:
    """Generate human-readable explanation"""
    
    parts = []
    
    # Novelty explanation
    if novelty > 0.8:
        parts.append("This represents a completely different perspective")
    elif novelty > 0.6:
        parts.append("This offers a fresh angle")
    else:
        parts.append("This provides a related but distinct viewpoint")
    
    # Connection explanation
    if connection > 0.7:
        parts.append("with strong conceptual links")
    elif connection > 0.5:
        parts.append("with meaningful connections")
    else:
        parts.append("with subtle but important relationships")
    
    # Shared concepts
    if shared_concepts:
        concepts_str = ", ".join(list(shared_concepts)[:3])
        parts.append(f"through shared concepts like {concepts_str}")
    
    return " ".join(parts)
```

**Test Command**:
```bash
# Test integrated endpoint
curl -X POST http://localhost:8000/api/v1/cognitive/query/enhanced \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What strategic decisions were made about expansion?",
    "project_id": "project-123",
    "include_bridges": true,
    "max_bridges": 3
  }'

# Test bridge explanation
curl -X GET "http://localhost:8000/api/v1/cognitive/bridges/explain/bridge-memory-id?query=strategic+decisions&activated_id=activated-memory-id"
```

**Performance Test**:
```python
# Load test with concurrent requests
import asyncio
import aiohttp
import time

async def test_request(session, query):
    async with session.post(
        "http://localhost:8000/api/v1/cognitive/query/enhanced",
        json={
            "query": query,
            "include_bridges": True
        }
    ) as response:
        result = await response.json()
        return result["processing_time_ms"]

async def load_test():
    queries = [
        "What are our strategic priorities?",
        "Who made commitments last week?",
        "What risks were identified?",
        "Show me innovation initiatives",
        "What did the CEO decide?"
    ]
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(20):  # 20 concurrent requests
            query = queries[i % len(queries)]
            tasks.append(test_request(session, query))
        
        start = time.time()
        times = await asyncio.gather(*tasks)
        total = time.time() - start
        
        print(f"Total time: {total:.2f}s")
        print(f"Average response: {np.mean(times):.0f}ms")
        print(f"Max response: {max(times):.0f}ms")
        
        # Should handle 20 concurrent requests in < 10s
        assert total < 10.0
        assert np.mean(times) < 2000  # Average < 2s

# Run: python -m pytest tests/test_load.py
```

**Success Criteria**:
- [ ] Activation and bridges work together
- [ ] Returns comprehensive results
- [ ] Bridge explanations are meaningful
- [ ] Handles concurrent requests
- [ ] Performance < 2s average

---

### Task 3.3.2: Implement Bridge Discovery Caching
**Time**: 1 hour
**Priority**: 🟡 Medium

**Location**: `src/cognitive/bridges/cache.py`

**Implementation**:
```python
import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Tuple
import asyncio
from dataclasses import dataclass, asdict
import numpy as np

from src.storage.sqlite_db import get_db_connection
from src.cognitive.bridges.engine import BridgeMemory

@dataclass
class CachedBridgeResult:
    """Cached bridge discovery result"""
    query_hash: str
    bridges: List[BridgeMemory]
    created_at: datetime
    expires_at: datetime
    hit_count: int = 0
    last_accessed: Optional[datetime] = None

class BridgeDiscoveryCache:
    """
    Persistent cache for bridge discovery results.
    Uses SQLite for storage and implements TTL-based expiration.
    """
    
    def __init__(self, default_ttl_hours: int = 24):
        self.default_ttl = timedelta(hours=default_ttl_hours)
        self._init_cache_table()
        self._cleanup_task = None
        
    def _init_cache_table(self):
        """Initialize cache table if not exists"""
        with get_db_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS bridge_discovery_cache (
                    query_hash TEXT PRIMARY KEY,
                    bridges_json TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    expires_at TIMESTAMP NOT NULL,
                    hit_count INTEGER DEFAULT 0,
                    last_accessed TIMESTAMP,
                    query_vector_sample TEXT,  -- First 10 dims for validation
                    activated_count INTEGER
                )
            """)
            
            # Create index for cleanup
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_bridge_cache_expires 
                ON bridge_discovery_cache(expires_at)
            """)
    
    async def get(self, 
                  query_vector: np.ndarray,
                  activated_memories: Set[str]) -> Optional[List[BridgeMemory]]:
        """
        Retrieve cached bridge results if available and valid.
        Updates hit count and last accessed time.
        """
        query_hash = self._generate_hash(query_vector, activated_memories)
        
        with get_db_connection() as conn:
            cursor = conn.execute("""
                SELECT bridges_json, expires_at, hit_count, query_vector_sample
                FROM bridge_discovery_cache
                WHERE query_hash = ? AND expires_at > ?
            """, (query_hash, datetime.now()))
            
            row = cursor.fetchone()
            if row:
                bridges_json, expires_at, hit_count, vector_sample = row
                
                # Validate vector hasn't changed significantly
                current_sample = self._vector_sample(query_vector)
                if vector_sample != current_sample:
                    # Vector changed, invalidate cache
                    self._invalidate(query_hash)
                    return None
                
                # Update statistics
                conn.execute("""
                    UPDATE bridge_discovery_cache
                    SET hit_count = ?, last_accessed = ?
                    WHERE query_hash = ?
                """, (hit_count + 1, datetime.now(), query_hash))
                
                # Deserialize bridges
                bridges_data = json.loads(bridges_json)
                bridges = [self._deserialize_bridge(b) for b in bridges_data]
                
                return bridges
        
        return None
    
    async def set(self,
                  query_vector: np.ndarray,
                  activated_memories: Set[str],
                  bridges: List[BridgeMemory],
                  ttl: Optional[timedelta] = None):
        """
        Cache bridge discovery results with TTL.
        """
        query_hash = self._generate_hash(query_vector, activated_memories)
        ttl = ttl or self.default_ttl
        
        # Serialize bridges
        bridges_data = [self._serialize_bridge(b) for b in bridges]
        bridges_json = json.dumps(bridges_data)
        
        # Store in cache
        with get_db_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO bridge_discovery_cache
                (query_hash, bridges_json, created_at, expires_at, 
                 query_vector_sample, activated_count)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                query_hash,
                bridges_json,
                datetime.now(),
                datetime.now() + ttl,
                self._vector_sample(query_vector),
                len(activated_memories)
            ))
    
    def _generate_hash(self, 
                      query_vector: np.ndarray,
                      activated_memories: Set[str]) -> str:
        """Generate stable hash for cache key"""
        
        # Use first 10 dimensions of vector + sorted memory IDs
        vector_key = "-".join(f"{x:.4f}" for x in query_vector[:10])
        
        # Sort activated memories for stable hash
        memory_key = "-".join(sorted(list(activated_memories)[:10]))
        
        # Create combined key
        combined = f"{vector_key}|{memory_key}"
        
        # Return SHA256 hash
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def _vector_sample(self, vector: np.ndarray) -> str:
        """Get vector sample for validation"""
        return "-".join(f"{x:.4f}" for x in vector[:10])
    
    def _serialize_bridge(self, bridge: BridgeMemory) -> Dict:
        """Serialize bridge for storage"""
        return {
            "memory_id": bridge.memory_id,
            "memory_content": bridge.memory_content,
            "bridge_score": bridge.bridge_score,
            "novelty_score": bridge.novelty_score,
            "connection_potential": bridge.connection_potential,
            "connection_path": bridge.connection_path,
            "explanation": bridge.explanation,
            "project_context": bridge.project_context,
            "stakeholder_relevance": bridge.stakeholder_relevance
        }
    
    def _deserialize_bridge(self, data: Dict) -> BridgeMemory:
        """Deserialize bridge from storage"""
        return BridgeMemory(**data)
    
    async def cleanup_expired(self) -> int:
        """Remove expired cache entries"""
        with get_db_connection() as conn:
            cursor = conn.execute("""
                DELETE FROM bridge_discovery_cache
                WHERE expires_at < ?
            """, (datetime.now(),))
            
            return cursor.rowcount
    
    async def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        with get_db_connection() as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_entries,
                    SUM(hit_count) as total_hits,
                    AVG(hit_count) as avg_hits,
                    MAX(hit_count) as max_hits,
                    COUNT(CASE WHEN expires_at > ? THEN 1 END) as active_entries
                FROM bridge_discovery_cache
            """, (datetime.now(),))
            
            row = cursor.fetchone()
            if row:
                return {
                    "total_entries": row[0] or 0,
                    "total_hits": row[1] or 0,
                    "avg_hits": row[2] or 0,
                    "max_hits": row[3] or 0,
                    "active_entries": row[4] or 0,
                    "cache_hit_rate": self._calculate_hit_rate()
                }
            
            return {}
    
    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate from logs"""
        # Simple implementation - in production, track this properly
        with get_db_connection() as conn:
            cursor = conn.execute("""
                SELECT SUM(hit_count) as hits, COUNT(*) as total
                FROM bridge_discovery_cache
                WHERE created_at > ?
            """, (datetime.now() - timedelta(hours=24),))
            
            row = cursor.fetchone()
            if row and row[1] > 0:
                # Approximate: assume each entry had at least one miss
                misses = row[1]
                hits = row[0] or 0
                total_requests = hits + misses
                return hits / total_requests if total_requests > 0 else 0.0
            
            return 0.0
    
    def _invalidate(self, query_hash: str):
        """Invalidate a specific cache entry"""
        with get_db_connection() as conn:
            conn.execute("""
                DELETE FROM bridge_discovery_cache
                WHERE query_hash = ?
            """, (query_hash,))
    
    async def start_cleanup_task(self, interval_hours: int = 1):
        """Start background cleanup task"""
        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(interval_hours * 3600)
                    deleted = await self.cleanup_expired()
                    if deleted > 0:
                        print(f"Cleaned up {deleted} expired bridge cache entries")
                except Exception as e:
                    print(f"Error in cache cleanup: {e}")
        
        self._cleanup_task = asyncio.create_task(cleanup_loop())
    
    def stop_cleanup_task(self):
        """Stop background cleanup task"""
        if self._cleanup_task:
            self._cleanup_task.cancel()

# Global cache instance
bridge_cache = BridgeDiscoveryCache()

# Decorator for easy caching
def cached_bridge_discovery(ttl_hours: int = 24):
    """Decorator to cache bridge discovery results"""
    def decorator(func):
        async def wrapper(self, query_vector: np.ndarray, 
                         activated_memories: Set[str], *args, **kwargs):
            # Try cache first
            cached = await bridge_cache.get(query_vector, activated_memories)
            if cached is not None:
                return cached
            
            # Call original function
            result = await func(self, query_vector, activated_memories, *args, **kwargs)
            
            # Cache result
            await bridge_cache.set(
                query_vector, 
                activated_memories, 
                result,
                ttl=timedelta(hours=ttl_hours)
            )
            
            return result
        
        return wrapper
    return decorator
```

**Test Command**:
```bash
# Create cache tests
cat > tests/test_bridge_cache.py << 'EOF'
import pytest
import numpy as np
from datetime import datetime, timedelta
from src.cognitive.bridges.cache import BridgeDiscoveryCache, BridgeMemory

@pytest.mark.asyncio
async def test_cache_operations():
    cache = BridgeDiscoveryCache(default_ttl_hours=1)
    
    # Test data
    query_vector = np.random.rand(400)
    activated = {"mem1", "mem2", "mem3"}
    bridges = [
        BridgeMemory(
            memory_id="bridge1",
            memory_content="Test bridge content",
            bridge_score=0.85,
            novelty_score=0.7,
            connection_potential=0.8,
            connection_path=["mem1", "bridge1"],
            explanation="Test explanation"
        )
    ]
    
    # Test set and get
    await cache.set(query_vector, activated, bridges)
    cached = await cache.get(query_vector, activated)
    
    assert cached is not None
    assert len(cached) == 1
    assert cached[0].memory_id == "bridge1"
    
    # Test cache miss with different vector
    different_vector = np.random.rand(400)
    cached_miss = await cache.get(different_vector, activated)
    assert cached_miss is None
    
    # Test expiration
    await cache.set(
        query_vector, 
        activated, 
        bridges, 
        ttl=timedelta(seconds=1)
    )
    
    await asyncio.sleep(2)
    expired = await cache.get(query_vector, activated)
    assert expired is None

@pytest.mark.asyncio
async def test_cache_stats():
    cache = BridgeDiscoveryCache()
    
    # Add some test data
    for i in range(5):
        vector = np.random.rand(400)
        activated = {f"mem{j}" for j in range(3)}
        bridges = [BridgeMemory(
            memory_id=f"bridge{i}",
            memory_content=f"Content {i}",
            bridge_score=0.8,
            novelty_score=0.7,
            connection_potential=0.7,
            connection_path=["mem1", f"bridge{i}"],
            explanation="Test"
        )]
        await cache.set(vector, activated, bridges)
    
    # Get stats
    stats = await cache.get_cache_stats()
    assert stats["total_entries"] >= 5
    assert stats["active_entries"] >= 5

@pytest.mark.asyncio 
async def test_cache_cleanup():
    cache = BridgeDiscoveryCache()
    
    # Add expired entry
    vector = np.random.rand(400)
    activated = {"mem1"}
    bridges = [BridgeMemory(
        memory_id="old_bridge",
        memory_content="Old content",
        bridge_score=0.7,
        novelty_score=0.6,
        connection_potential=0.6,
        connection_path=["mem1", "old_bridge"],
        explanation="Old"
    )]
    
    await cache.set(vector, activated, bridges, ttl=timedelta(seconds=-1))
    
    # Run cleanup
    deleted = await cache.cleanup_expired()
    assert deleted >= 1
    
    # Verify it's gone
    cached = await cache.get(vector, activated)
    assert cached is None
EOF

pytest tests/test_bridge_cache.py -v
```

**Integration with Engine**:
```python
# Update BridgeDiscoveryEngine to use cache
from src.cognitive.bridges.cache import cached_bridge_discovery

class BridgeDiscoveryEngine:
    # ... existing code ...
    
    @cached_bridge_discovery(ttl_hours=24)
    async def discover_bridges(self, 
                             query_vector: np.ndarray,
                             activated_memories: Set[str],
                             context: Optional[Dict] = None) -> List[BridgeMemory]:
        # Existing implementation
        # Cache decorator handles caching automatically
        pass
```

**Success Criteria**:
- [ ] Cache stores and retrieves correctly
- [ ] TTL expiration works
- [ ] Cache stats are accurate
- [ ] Cleanup removes expired entries
- [ ] Integration with engine works

---

### Task 3.3.3: Create Performance Monitoring
**Time**: 45 minutes
**Priority**: 🟡 Medium

**Location**: `src/monitoring/performance.py`

**Implementation**:
```python
import time
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
from contextlib import asynccontextmanager

@dataclass
class PerformanceMetric:
    """Single performance measurement"""
    operation: str
    duration_ms: float
    timestamp: datetime
    success: bool
    metadata: Dict = field(default_factory=dict)

@dataclass 
class PerformanceStats:
    """Aggregated performance statistics"""
    operation: str
    count: int
    total_ms: float
    min_ms: float
    max_ms: float
    avg_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    success_rate: float
    
class PerformanceMonitor:
    """Monitor and track performance metrics"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.metrics: Dict[str, List[PerformanceMetric]] = defaultdict(list)
        self._lock = asyncio.Lock()
        
    @asynccontextmanager
    async def track(self, operation: str, **metadata):
        """Context manager to track operation performance"""
        start = time.time()
        success = True
        
        try:
            yield
        except Exception as e:
            success = False
            metadata['error'] = str(e)
            raise
        finally:
            duration = (time.time() - start) * 1000
            
            async with self._lock:
                metric = PerformanceMetric(
                    operation=operation,
                    duration_ms=duration,
                    timestamp=datetime.now(),
                    success=success,
                    metadata=metadata
                )
                
                # Add to metrics
                self.metrics[operation].append(metric)
                
                # Keep window size
                if len(self.metrics[operation]) > self.window_size:
                    self.metrics[operation] = self.metrics[operation][-self.window_size:]
    
    def get_stats(self, operation: str) -> Optional[PerformanceStats]:
        """Get performance statistics for an operation"""
        if operation not in self.metrics:
            return None
        
        metrics = self.metrics[operation]
        if not metrics:
            return None
        
        # Extract durations
        durations = [m.duration_ms for m in metrics]
        successful = [m for m in metrics if m.success]
        
        # Calculate percentiles
        sorted_durations = sorted(durations)
        
        return PerformanceStats(
            operation=operation,
            count=len(metrics),
            total_ms=sum(durations),
            min_ms=min(durations),
            max_ms=max(durations),
            avg_ms=np.mean(durations),
            p50_ms=np.percentile(sorted_durations, 50),
            p95_ms=np.percentile(sorted_durations, 95),
            p99_ms=np.percentile(sorted_durations, 99),
            success_rate=len(successful) / len(metrics)
        )
    
    def get_all_stats(self) -> Dict[str, PerformanceStats]:
        """Get stats for all tracked operations"""
        return {
            op: self.get_stats(op) 
            for op in self.metrics.keys()
        }
    
    def get_recent_metrics(self, 
                          operation: str, 
                          limit: int = 10) -> List[PerformanceMetric]:
        """Get recent metrics for an operation"""
        if operation not in self.metrics:
            return []
        
        return self.metrics[operation][-limit:]
    
    def clear_metrics(self, operation: Optional[str] = None):
        """Clear metrics for specific operation or all"""
        if operation:
            self.metrics[operation].clear()
        else:
            self.metrics.clear()

# Global monitor instance
perf_monitor = PerformanceMonitor()

# Decorator for performance tracking
def track_performance(operation: str):
    """Decorator to track function performance"""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                async with perf_monitor.track(operation):
                    return await func(*args, **kwargs)
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                start = time.time()
                success = True
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    success = False
                    raise
                finally:
                    duration = (time.time() - start) * 1000
                    # Can't use async in sync function, so just record
                    metric = PerformanceMetric(
                        operation=operation,
                        duration_ms=duration,
                        timestamp=datetime.now(),
                        success=success,
                        metadata={}
                    )
                    perf_monitor.metrics[operation].append(metric)
            return sync_wrapper
    return decorator

# Phase 3 specific monitors
class Phase3PerformanceTargets:
    """Performance targets for Phase 3 operations"""
    
    TARGETS = {
        "bridge_discovery": {
            "p95_ms": 1000,  # 95th percentile < 1s
            "p99_ms": 1500,  # 99th percentile < 1.5s
            "success_rate": 0.99
        },
        "dimension_extraction": {
            "p95_ms": 50,   # Very fast
            "p99_ms": 100,
            "success_rate": 0.999
        },
        "integrated_query": {
            "p95_ms": 2000,  # Full pipeline < 2s
            "p99_ms": 3000,
            "success_rate": 0.99
        }
    }
    
    @classmethod
    def check_targets(cls) -> Dict[str, Dict[str, bool]]:
        """Check if performance targets are met"""
        results = {}
        
        for operation, targets in cls.TARGETS.items():
            stats = perf_monitor.get_stats(operation)
            if not stats:
                results[operation] = {"no_data": True}
                continue
            
            results[operation] = {
                "p95_ok": stats.p95_ms <= targets["p95_ms"],
                "p99_ok": stats.p99_ms <= targets["p99_ms"],
                "success_ok": stats.success_rate >= targets["success_rate"],
                "stats": {
                    "p95_ms": stats.p95_ms,
                    "p99_ms": stats.p99_ms,
                    "success_rate": stats.success_rate
                }
            }
        
        return results
```

**Test Command**:
```bash
# Create performance monitoring tests
cat > tests/test_performance_monitoring.py << 'EOF'
import pytest
import asyncio
from src.monitoring.performance import (
    PerformanceMonitor, 
    track_performance,
    Phase3PerformanceTargets
)

@pytest.mark.asyncio
async def test_performance_tracking():
    monitor = PerformanceMonitor()
    
    # Track some operations
    async with monitor.track("test_operation"):
        await asyncio.sleep(0.1)
    
    async with monitor.track("test_operation"):
        await asyncio.sleep(0.05)
    
    # Get stats
    stats = monitor.get_stats("test_operation")
    assert stats is not None
    assert stats.count == 2
    assert stats.min_ms >= 50
    assert stats.max_ms >= 100
    assert stats.success_rate == 1.0

@pytest.mark.asyncio
async def test_performance_decorator():
    @track_performance("decorated_function")
    async def slow_function():
        await asyncio.sleep(0.1)
        return "done"
    
    # Call function
    result = await slow_function()
    assert result == "done"
    
    # Check metrics were recorded
    from src.monitoring.performance import perf_monitor
    stats = perf_monitor.get_stats("decorated_function")
    assert stats is not None
    assert stats.count >= 1
    assert stats.avg_ms >= 100

def test_performance_targets():
    # Check targets structure
    targets = Phase3PerformanceTargets.TARGETS
    assert "bridge_discovery" in targets
    assert "dimension_extraction" in targets
    assert "integrated_query" in targets
    
    # Targets should be reasonable
    assert targets["bridge_discovery"]["p95_ms"] == 1000
    assert targets["integrated_query"]["p95_ms"] == 2000
EOF

pytest tests/test_performance_monitoring.py -v
```

**Integration Example**:
```python
# Add to bridge discovery engine
class BridgeDiscoveryEngine:
    @track_performance("bridge_discovery")
    async def discover_bridges(self, ...):
        # Existing implementation
        pass

# Add to dimension extractors
class SocialDimensionExtractor:
    @track_performance("dimension_extraction.social")
    def extract(self, text: str) -> np.ndarray:
        # Existing implementation
        pass

# Add monitoring endpoint
@router.get("/api/v1/monitoring/performance")
async def get_performance_stats():
    """Get current performance statistics"""
    all_stats = perf_monitor.get_all_stats()
    target_checks = Phase3PerformanceTargets.check_targets()
    
    return {
        "stats": {
            op: {
                "count": stats.count,
                "avg_ms": stats.avg_ms,
                "p95_ms": stats.p95_ms,
                "p99_ms": stats.p99_ms,
                "success_rate": stats.success_rate
            }
            for op, stats in all_stats.items()
        },
        "target_compliance": target_checks,
        "timestamp": datetime.now()
    }
```

**Success Criteria**:
- [ ] Tracks operation performance
- [ ] Calculates accurate statistics
- [ ] Decorator works correctly
- [ ] Performance targets defined
- [ ] Can check target compliance

---

## Phase 3 Completion Checklist

### Core Implementation
- [ ] Bridge Discovery Engine implemented
- [ ] Distance inversion algorithm working
- [ ] Bridge explanations generated
- [ ] Social dimension extractor enhanced
- [ ] Causal dimension extractor implemented
- [ ] Strategic dimension extractor created
- [ ] All extractors return normalized 3D vectors

### Integration
- [ ] Bridge discovery integrated with activation
- [ ] Enhanced API endpoints created
- [ ] Explanation endpoint working
- [ ] All dimensions integrated in 400D vector

### Performance
- [ ] Bridge discovery < 1s for 5 bridges
- [ ] Dimension extraction < 50ms each
- [ ] Full query pipeline < 2s
- [ ] Caching implemented and working
- [ ] Performance monitoring active

### Quality
- [ ] All tests passing
- [ ] Type hints complete
- [ ] Error handling robust
- [ ] Logging comprehensive
- [ ] Documentation updated

### Consulting Features
- [ ] Strategic alignment detection
- [ ] Stakeholder extraction
- [ ] Initiative identification
- [ ] Risk/opportunity balance
- [ ] Project context respected

## Next Phase Preview
Phase 4 will implement:
- Memory consolidation with DBSCAN
- Semantic memory generation
- Lifecycle management
- Background task scheduling
- Parent-child relationships

## Commands Summary
```bash
# Run all Phase 3 tests
pytest tests/test_bridge_*.py tests/test_*_dimension.py -v

# Check performance
curl http://localhost:8000/api/v1/monitoring/performance

# Test integrated query
curl -X POST http://localhost:8000/api/v1/cognitive/query/enhanced \
  -H "Content-Type: application/json" \
  -d '{"query": "strategic decisions", "include_bridges": true}'

# Verify all dimensions
python -c "
from src.features.dimensions import *
text = 'The CEO decided we must expand into new markets by Q2'
print('Social:', social_extractor.extract(text))
print('Causal:', causal_extractor.extract(text))
print('Strategic:', strategic_extractor.extract(text))
"
```
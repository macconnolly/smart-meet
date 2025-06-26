# API Endpoints Documentation

## Base URL
```
Development: http://localhost:8000
Production: https://api.cognitive-meeting.example.com
```

## Authentication
```
Authorization: Bearer <JWT_TOKEN>
Content-Type: application/json
```

## Phase 1 Endpoints (Basic Operations)

### 1. Health Check
```http
GET /health
```
**Response:**
```json
{
  "status": "healthy",
  "phase": 1,
  "version": "1.0.0"
}
```

### 2. Ingest Meeting
```http
POST /ingest
```
**Request:**
```json
{
  "transcript": "John: We decided to implement caching...",
  "metadata": {
    "title": "Engineering Planning",
    "participants": ["John", "Mary", "Tom"],
    "start_time": "2024-01-15T10:00:00Z",
    "end_time": "2024-01-15T11:00:00Z",
    "platform": "zoom",
    "series_id": "weekly-eng"
  }
}
```
**Response:**
```json
{
  "meeting_id": "550e8400-e29b-41d4-a716-446655440000",
  "title": "Engineering Planning",
  "memory_count": 15,
  "status": "success",
  "processing_time_ms": 1234.5
}
```

### 3. Simple Search
```http
POST /search
```
**Request:**
```json
{
  "query": "caching decisions",
  "limit": 10
}
```
**Response:**
```json
{
  "results": [
    {
      "id": "mem-123",
      "content": "We decided to implement Redis caching",
      "speaker": "John",
      "content_type": "decision",
      "score": 0.95,
      "meeting_id": "550e8400-e29b-41d4-a716-446655440000",
      "timestamp": 125.5
    }
  ],
  "count": 10,
  "processing_time_ms": 156.7
}
```

## Phase 2 Endpoints (Cognitive Features)

### 1. Cognitive Query
```http
POST /api/v2/query
```
**Request:**
```json
{
  "query": "What performance optimizations were discussed?",
  "enable_activation": true,
  "enable_bridges": true,
  "max_results": 20,
  "include_semantic": true,
  "include_episodic": true
}
```
**Response:**
```json
{
  "direct_results": [
    {
      "id": "mem-123",
      "content": "We need to optimize database queries",
      "speaker": "Tom",
      "meeting_id": "550e8400-e29b-41d4-a716-446655440000",
      "timestamp": 245.5,
      "memory_type": "episodic",
      "content_type": "insight",
      "level": 2,
      "importance_score": 0.85,
      "access_count": 12,
      "created_at": "2024-01-15T10:04:05Z"
    }
  ],
  "activated_memories": {
    "core": [
      {
        "memory": { /* Memory object */ },
        "activation_level": 0.95,
        "depth": 0,
        "path": ["mem-123"]
      }
    ],
    "contextual": [
      {
        "memory": { /* Memory object */ },
        "activation_level": 0.82,
        "depth": 1,
        "path": ["mem-123", "mem-456"]
      }
    ],
    "peripheral": [
      {
        "memory": { /* Memory object */ },
        "activation_level": 0.71,
        "depth": 2,
        "path": ["mem-123", "mem-456", "mem-789"]
      }
    ],
    "total_activated": 15,
    "max_depth_reached": 3
  },
  "bridge_memories": [
    {
      "memory": {
        "id": "mem-999",
        "content": "Database indexing strategies improve query performance",
        "speaker": "Mary",
        "memory_type": "semantic"
      },
      "bridge_score": 0.78,
      "novelty_score": 0.85,
      "connection_strength": 0.71,
      "explanation": "Novel perspective (0.85) with strong connection (0.71) to activated memory"
    }
  ],
  "processing_breakdown": {
    "embedding_ms": 45.2,
    "search_ms": 156.7,
    "activation_ms": 234.5,
    "bridge_ms": 456.8,
    "total_ms": 893.2
  },
  "total_time_ms": 893.2
}
```

### 2. Get Meeting Memories
```http
GET /api/v1/meetings/{meeting_id}/memories
```
**Response:**
```json
{
  "memories": [
    {
      "id": "mem-123",
      "content": "We decided to implement caching",
      "speaker": "John",
      "timestamp": 125.5,
      "memory_type": "episodic",
      "content_type": "decision",
      "level": 2,
      "importance_score": 0.9,
      "access_count": 5,
      "created_at": "2024-01-15T10:02:05Z"
    }
  ],
  "count": 25,
  "meeting": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "title": "Engineering Planning",
    "start_time": "2024-01-15T10:00:00Z",
    "end_time": "2024-01-15T11:00:00Z"
  }
}
```

### 3. Trigger Consolidation
```http
POST /api/v2/consolidate
```
**Response:**
```json
{
  "status": "Consolidation triggered",
  "message": "Running in background",
  "estimated_time_seconds": 120
}
```

### 4. Get Performance Statistics
```http
GET /api/v2/stats/performance
```
**Response:**
```json
{
  "avg_query_time_ms": 856.3,
  "total_memories": 12543,
  "semantic_memories": 1254,
  "episodic_memories": 11289,
  "total_connections": 45678,
  "cache_hit_rate": 0.72,
  "last_consolidation": "2024-01-15T08:00:00Z",
  "performance_by_operation": {
    "embedding_generation": {
      "avg_ms": 45.2,
      "p95_ms": 78.5,
      "p99_ms": 95.3
    },
    "similarity_search": {
      "avg_ms": 156.7,
      "p95_ms": 234.5,
      "p99_ms": 345.6
    },
    "activation_spreading": {
      "avg_ms": 234.5,
      "p95_ms": 456.7,
      "p99_ms": 567.8
    },
    "bridge_discovery": {
      "avg_ms": 456.8,
      "p95_ms": 789.0,
      "p99_ms": 890.1
    }
  }
}
```

### 5. Health Check (Enhanced)
```http
GET /api/v2/health
```
**Response:**
```json
{
  "status": "healthy",
  "checks": {
    "api": "healthy",
    "database": "healthy",
    "qdrant": "healthy",
    "consolidation_scheduler": "running",
    "lifecycle_manager": "running"
  },
  "version": "2.0.0",
  "phase": 2,
  "uptime_seconds": 345600,
  "memory_usage_mb": 1234.5
}
```

## Advanced Query Parameters

### Filtering Options
```json
{
  "query": "performance optimization",
  "filters": {
    "meeting_ids": ["meeting-123", "meeting-456"],
    "content_types": ["decision", "action"],
    "memory_types": ["semantic"],
    "speakers": ["John", "Mary"],
    "date_range": {
      "start": "2024-01-01T00:00:00Z",
      "end": "2024-01-31T23:59:59Z"
    },
    "min_importance": 0.7
  }
}
```

### Activation Control
```json
{
  "query": "caching strategy",
  "activation_config": {
    "threshold": 0.8,
    "max_depth": 3,
    "max_activations": 30,
    "decay_factor": 0.75,
    "prefer_semantic": true
  }
}
```

### Bridge Discovery Control
```json
{
  "query": "database optimization",
  "bridge_config": {
    "novelty_weight": 0.7,
    "connection_weight": 0.3,
    "min_novelty": 0.6,
    "max_bridges": 10,
    "exclude_speakers": ["Bot"]
  }
}
```

## Error Responses

### 400 Bad Request
```json
{
  "error": "Invalid request",
  "message": "Query parameter is required",
  "field": "query"
}
```

### 404 Not Found
```json
{
  "error": "Resource not found",
  "message": "Meeting with ID '123' not found"
}
```

### 500 Internal Server Error
```json
{
  "error": "Internal server error",
  "message": "Failed to generate embeddings",
  "request_id": "req-abc123",
  "timestamp": "2024-01-15T10:00:00Z"
}
```

### 503 Service Unavailable
```json
{
  "error": "Service unavailable",
  "message": "Qdrant connection failed",
  "retry_after_seconds": 30
}
```

## Rate Limiting
- 1000 requests per hour per API key
- 100 concurrent requests maximum
- Headers:
  - `X-RateLimit-Limit`: 1000
  - `X-RateLimit-Remaining`: 950
  - `X-RateLimit-Reset`: 1705320000

## Pagination
For endpoints returning lists:
```json
{
  "results": [...],
  "pagination": {
    "total": 1234,
    "page": 1,
    "per_page": 20,
    "total_pages": 62
  },
  "links": {
    "first": "/api/v2/memories?page=1",
    "prev": null,
    "next": "/api/v2/memories?page=2",
    "last": "/api/v2/memories?page=62"
  }
}
```

## WebSocket Endpoint (Future)
```
ws://localhost:8000/ws/realtime
```
For real-time memory updates during meetings.
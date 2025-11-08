# API Documentation

This document describes the CRISPR Design Agent API, including both basic and extended versions.

## Overview

The API provides two versions:
- **Basic API** (`api/app.py`): Simple scoring and design endpoints
- **Extended API** (`api/app_extended.py`): Production-ready with batch scoring, audit logging, and rate limiting

## Quick Start

### Basic API

```bash
# Start server
uvicorn api.app:app --host 0.0.0.0 --port 8000

# Test health endpoint
curl http://localhost:8000/health
```

### Extended API

```bash
# Set environment variables (optional)
export CHECKPOINT_PATH=models/checkpoints/best.ckpt
export AUDIT_LOG_DIR=logs/audit
export DEVICE=cuda
export ALLOWED_ORIGINS="https://myapp.com,https://api.myapp.com"

# Start server
uvicorn api.app_extended:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### Health Check

**GET** `/health`

Check API health and model status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda"
}
```

### Single Sequence Scoring

**POST** `/score`

Score a single protein sequence for a specific task.

**Rate Limit:** 100 requests/minute

**Request:**
```json
{
  "sequence": "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSP",
  "task": "clinvar",
  "user_id": "user123"
}
```

**Response:**
```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "task": "clinvar",
  "score": 0.85,
  "confidence": 0.85,
  "metadata": {
    "sequence_length": 48
  }
}
```

**Available Tasks:**
- `dms`: Deep Mutational Scanning effect prediction
- `depmap`: DepMap gene essentiality prediction
- `clinvar`: ClinVar pathogenicity classification

**Errors:**
- `400 Bad Request`: Invalid sequence or task
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Model inference error

### Batch Sequence Scoring

**POST** `/batch-score` *(Extended API only)*

Score multiple sequences in a single request.

**Rate Limit:** 10 requests/minute

**Request:**
```json
{
  "sequences": [
    "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSP",
    "MDGTALVSLCGLCGAVGEAKMQKLLEEQRREERAQREQAQRKEKKLV",
    "MSRGVTSTTGNIRRNPDRGRKTPPTPAQLFNLLWKTGSGLEKMNELK"
  ],
  "task": "dms",
  "user_id": "user123"
}
```

**Response:**
```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "task": "dms",
  "results": [
    {
      "sequence_index": 0,
      "score": 0.72,
      "confidence": 0.72,
      "sequence_length": 48
    },
    {
      "sequence_index": 1,
      "score": 0.65,
      "confidence": 0.65,
      "sequence_length": 48
    },
    {
      "sequence_index": 2,
      "score": 0.81,
      "confidence": 0.81,
      "sequence_length": 48
    }
  ],
  "metadata": {
    "num_sequences": 3
  }
}
```

**Constraints:**
- Minimum: 1 sequence
- Maximum: 100 sequences per request
- Each sequence max length: 10,000 amino acids

### API Statistics

**GET** `/stats` *(Extended API only)*

Get API usage statistics from audit logs.

**Rate Limit:** 10 requests/minute

**Response:**
```json
{
  "total_requests": 1523,
  "endpoints": {
    "/score": 1234,
    "/batch-score": 289
  },
  "status_codes": {
    "200": 1489,
    "400": 24,
    "429": 10
  },
  "avg_duration_ms": 145.3,
  "errors": 34
}
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CHECKPOINT_PATH` | None | Path to model checkpoint |
| `AUDIT_LOG_DIR` | `logs/audit` | Directory for audit logs |
| `DEVICE` | `auto` | Device for inference (`cuda`, `cpu`, `auto`) |
| `ALLOWED_ORIGINS` | `*` | CORS allowed origins (comma-separated) |
| `HASH_SEQUENCES` | `true` | Hash sequences in audit logs for privacy |

### Rate Limiting

Rate limits are applied per IP address:
- Single score: 100 requests/minute
- Batch score: 10 requests/minute
- Stats: 10 requests/minute

To adjust limits, modify the `@limiter.limit()` decorators in `api/app_extended.py`.

## Audit Logging

The extended API automatically logs all requests to `AUDIT_LOG_DIR` in JSONL format.

**Audit Entry Format:**
```json
{
  "timestamp": "2025-01-15T10:30:45.123456",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "endpoint": "/score",
  "method": "POST",
  "user_id": "user123",
  "request_data": {
    "sequence_hash": "a1b2c3d4e5f6g7h8",
    "sequence_length": 48,
    "task": "clinvar"
  },
  "response_data": {
    "score": 0.85,
    "confidence": 0.85
  },
  "status_code": 200,
  "duration_ms": 142.5,
  "error": null,
  "metadata": {}
}
```

**Privacy:**
- Sequences are hashed by default (configurable via `HASH_SEQUENCES`)
- Only hash and length are stored
- User IDs are optional and controlled by client

**Log Rotation:**
- New file created daily: `audit_YYYYMMDD.jsonl`
- No automatic cleanup - implement your own retention policy

## Client Examples

### Python

```python
import requests

# Single score
response = requests.post(
    "http://localhost:8000/score",
    json={
        "sequence": "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSP",
        "task": "clinvar",
        "user_id": "user123"
    }
)
print(response.json())

# Batch score
response = requests.post(
    "http://localhost:8000/batch-score",
    json={
        "sequences": [
            "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSP",
            "MDGTALVSLCGLCGAVGEAKMQKLLEEQRREERAQREQAQRKEKKLV"
        ],
        "task": "dms",
        "user_id": "user123"
    }
)
print(response.json())
```

### cURL

```bash
# Single score
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "sequence": "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSP",
    "task": "clinvar",
    "user_id": "user123"
  }'

# Batch score
curl -X POST http://localhost:8000/batch-score \
  -H "Content-Type: application/json" \
  -d '{
    "sequences": [
      "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSP",
      "MDGTALVSLCGLCGAVGEAKMQKLLEEQRREERAQREQAQRKEKKLV"
    ],
    "task": "dms"
  }'
```

### JavaScript

```javascript
// Single score
const response = await fetch('http://localhost:8000/score', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    sequence: 'MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSP',
    task: 'clinvar',
    user_id: 'user123'
  })
});
const data = await response.json();
console.log(data);

// Batch score
const batchResponse = await fetch('http://localhost:8000/batch-score', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    sequences: [
      'MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSP',
      'MDGTALVSLCGLCGAVGEAKMQKLLEEQRREERAQREQAQRKEKKLV'
    ],
    task: 'dms'
  })
});
const batchData = await batchResponse.json();
console.log(batchData);
```

## Deployment

### Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV CHECKPOINT_PATH=models/checkpoints/best.ckpt
ENV DEVICE=cuda

CMD ["uvicorn", "api.app_extended:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build and run
docker build -t crispr-api .
docker run -p 8000:8000 -v ./models:/app/models crispr-api
```

### Production Considerations

1. **Use HTTPS** with reverse proxy (nginx/traefik)
2. **Set ALLOWED_ORIGINS** to specific domains
3. **Monitor audit logs** for suspicious activity
4. **Scale horizontally** with load balancer
5. **Implement authentication** if exposing publicly
6. **Set up log rotation** and backup
7. **Monitor GPU memory** usage
8. **Use connection pooling** for database (if added)

## Error Handling

### Common Errors

**400 Bad Request**
```json
{
  "detail": "Sequence must contain only amino acid letters"
}
```

**429 Too Many Requests**
```json
{
  "detail": "Rate limit exceeded: 100 per 1 minute"
}
```

**500 Internal Server Error**
```json
{
  "detail": "Model inference failed: CUDA out of memory"
}
```

### Best Practices

1. **Implement retry logic** with exponential backoff
2. **Handle rate limits** gracefully
3. **Validate sequences** on client side
4. **Batch requests** when scoring multiple sequences
5. **Monitor request_id** for debugging
6. **Check health endpoint** before making requests

## API Versioning

Current version: `2.0.0`

Version information available at:
- OpenAPI docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Validation](https://docs.pydantic.dev/)
- [SlowAPI Rate Limiting](https://slowapi.readthedocs.io/)

# Deep RAG API Documentation

## Overview

The Deep RAG API is a FastAPI-based REST API that provides access to the Deep RAG (Retrieval-Augmented Generation) system for complex question answering with multi-step reasoning.

## Base URL

- **Local Development**: `http://localhost:8000`
- **Production**: `https://<your-container-app-fqdn>`

## API Endpoints

### Health Check

**GET** `/health`

Check the health status of the API and Deep RAG service.

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "timestamp": "2024-01-15T10:30:00Z",
  "system_info": {
    "initialized": true,
    "embeddings_loaded": true,
    "vector_store_ready": true
  }
}
```

**Status Values:**
- `healthy`: Service is ready to handle requests
- `degraded`: Service is partially ready
- `unhealthy`: Service is not ready

---

### Query (Synchronous)

**POST** `/api/v1/query`

Answer a question using the Deep RAG system. Returns the complete answer after processing.

**Request Body:**
```json
{
  "question": "What are the key cost benchmarks for green hydrogen production in India?",
  "max_steps": 5,
  "temperature": 0.0
}
```

**Parameters:**
- `question` (required, string): The question to answer
- `max_steps` (optional, integer): Maximum reasoning steps (1-20, default: 7)
- `temperature` (optional, float): LLM temperature (0.0-2.0, default: from config)

**Response:**
```json
{
  "answer": "Green hydrogen cost benchmarks in India range from $2.50 to $3.00 per kilogram...",
  "question": "What are the key cost benchmarks for green hydrogen production in India?",
  "steps_taken": 3,
  "sources": [
    {
      "content": "Document snippet...",
      "source": "document.pdf",
      "metadata": {}
    }
  ],
  "processing_time": 12.5,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

**Status Codes:**
- `200 OK`: Success
- `400 Bad Request`: Invalid request
- `503 Service Unavailable`: Service not ready
- `500 Internal Server Error`: Processing error

---

### Query (Streaming)

**POST** `/api/v1/query/stream`

Answer a question using the Deep RAG system with real-time streaming updates via Server-Sent Events (SSE).

**Request Body:**
```json
{
  "question": "What are the key cost benchmarks for green hydrogen production in India?",
  "max_steps": 5,
  "temperature": 0.0
}
```

**Response Format:** Server-Sent Events (SSE)

**Event Types:**
- `plan`: Plan generation event
- `retrieval`: Document retrieval event
- `reflection`: Reflection/summary event
- `answer`: Final answer event
- `complete`: Processing complete
- `error`: Error event

**Example Stream:**
```
data: {"type":"plan","content":"Generated plan with 3 steps","step":0,"metadata":{"steps":["..."],"total_steps":3},"timestamp":"2024-01-15T10:30:00Z"}

data: {"type":"retrieval","content":"Retrieved 5 documents for step 1","step":1,"metadata":{"doc_count":5},"timestamp":"2024-01-15T10:30:01Z"}

data: {"type":"reflection","content":"Step 1 reflection: ...","step":1,"metadata":{},"timestamp":"2024-01-15T10:30:02Z"}

data: {"type":"answer","content":"Green hydrogen cost benchmarks...","step":3,"metadata":{"answer_length":500},"timestamp":"2024-01-15T10:30:05Z"}

data: {"type":"complete","content":"Processing complete","step":3,"metadata":{"total_steps":3},"timestamp":"2024-01-15T10:30:05Z"}

data: [DONE]
```

**Status Codes:**
- `200 OK`: Stream started
- `400 Bad Request`: Invalid request
- `503 Service Unavailable`: Service not ready
- `500 Internal Server Error`: Processing error

---

## Usage Examples

### Python

#### Synchronous Request

```python
import requests

url = "https://your-api-url/api/v1/query"
payload = {
    "question": "What are green hydrogen cost benchmarks?",
    "max_steps": 5
}

response = requests.post(url, json=payload)
data = response.json()

print(f"Answer: {data['answer']}")
print(f"Processing time: {data['processing_time']}s")
```

#### Streaming Request

```python
import requests
import json

url = "https://your-api-url/api/v1/query/stream"
payload = {
    "question": "What are green hydrogen cost benchmarks?",
    "max_steps": 5
}

response = requests.post(url, json=payload, stream=True)

for line in response.iter_lines():
    if line:
        line_str = line.decode('utf-8')
        if line_str.startswith('data: '):
            data_str = line_str[6:]  # Remove 'data: ' prefix
            if data_str == '[DONE]':
                break
            try:
                chunk = json.loads(data_str)
                print(f"[{chunk['type']}] {chunk['content'][:100]}...")
            except json.JSONDecodeError:
                pass
```

### cURL

#### Synchronous Request

```bash
curl -X POST "https://your-api-url/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are green hydrogen cost benchmarks?",
    "max_steps": 5
  }'
```

#### Streaming Request

```bash
curl -X POST "https://your-api-url/api/v1/query/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are green hydrogen cost benchmarks?",
    "max_steps": 5
  }' \
  --no-buffer
```

### JavaScript/TypeScript

#### Synchronous Request

```javascript
const response = await fetch('https://your-api-url/api/v1/query', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    question: 'What are green hydrogen cost benchmarks?',
    max_steps: 5
  })
});

const data = await response.json();
console.log('Answer:', data.answer);
```

#### Streaming Request

```javascript
const response = await fetch('https://your-api-url/api/v1/query/stream', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    question: 'What are green hydrogen cost benchmarks?',
    max_steps: 5
  })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  
  const chunk = decoder.decode(value);
  const lines = chunk.split('\n');
  
  for (const line of lines) {
    if (line.startsWith('data: ')) {
      const dataStr = line.slice(6);
      if (dataStr === '[DONE]') {
        return;
      }
      try {
        const chunk = JSON.parse(dataStr);
        console.log(`[${chunk.type}]`, chunk.content);
      } catch (e) {
        // Ignore parse errors
      }
    }
  }
}
```

## Error Handling

### Error Response Format

```json
{
  "error": "Error message",
  "error_type": "ErrorType",
  "timestamp": "2024-01-15T10:30:00Z",
  "details": {
    "additional": "information"
  }
}
```

### Common Error Codes

- `400 Bad Request`: Invalid request parameters
- `422 Unprocessable Entity`: Validation error
- `503 Service Unavailable`: Service not ready (check `/health`)
- `504 Gateway Timeout`: Processing timeout
- `500 Internal Server Error`: Server error

## Rate Limiting

Currently, rate limiting is handled by Azure Container Apps scaling. For production use, consider implementing:

- API key authentication
- Rate limiting middleware
- Request queuing

## Authentication

Currently, the API does not require authentication. For production deployments, consider:

- API key authentication
- Azure AD integration
- OAuth 2.0

## Interactive API Documentation

FastAPI provides interactive API documentation:

- **Swagger UI**: `https://your-api-url/docs`
- **ReDoc**: `https://your-api-url/redoc`

## Testing

A comprehensive test script is provided to test all API endpoints:

```bash
# Test all endpoints
python scripts/test_api.py

# Test with custom question
python scripts/test_api.py --question "What are green hydrogen cost benchmarks?"

# Test only health endpoint
python scripts/test_api.py --health-only

# Test only streaming
python scripts/test_api.py --stream-only

# Test Azure deployment
python scripts/test_api.py --base-url https://your-app.azurecontainerapps.io
```

The test script provides:
- Colored output for easy reading
- Detailed response information
- Error handling and reporting
- Test summary with pass/fail status

See `scripts/test_api.py --help` for all available options.

## Support

For issues or questions:

1. Check the health endpoint: `/health`
2. Review logs in Azure Portal
3. Check the deployment documentation in `azure/README.md`
4. Run the test script to verify API functionality


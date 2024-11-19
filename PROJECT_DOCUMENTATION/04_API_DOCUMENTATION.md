# Mino AI - API Documentation

## API Overview
Mino AI provides a RESTful API for chat interactions and AI processing. The API is built using Flask and integrates with Google's Generative AI.

## Base URL
- Local Development: `http://localhost:10000`
- Production: `https://mino-ai.onrender.com`

## Endpoints

### 1. Chat Endpoint
```
POST /chat
```

#### Request
- Content-Type: `application/json`
```json
{
    "message": "string",
    "context": "string" (optional)
}
```

#### Response
```json
{
    "response": "string",
    "emotion": "string",
    "confidence": float
}
```

#### Example
```python
import requests

url = "https://mino-ai.onrender.com/chat"
payload = {
    "message": "Hello, how are you?",
    "context": "Casual greeting"
}
response = requests.post(url, json=payload)
print(response.json())
```

### 2. Health Check
```
GET /health
```

#### Response
```json
{
    "status": "healthy",
    "timestamp": "ISO-8601 timestamp"
}
```

## Error Handling

### Error Responses
```json
{
    "error": "string",
    "code": "string",
    "details": "string" (optional)
}
```

### Common Error Codes
- 400: Bad Request
- 401: Unauthorized
- 429: Too Many Requests
- 500: Internal Server Error

## Rate Limiting
- 60 requests per minute per IP
- 1000 requests per day per IP

## Authentication
Currently, no authentication is required for API access. However, the server uses API keys for Google Generative AI integration.

## Response Formats

### Success Response
```json
{
    "response": "AI generated response",
    "emotion": "detected emotion",
    "confidence": 0.95,
    "timestamp": "2024-02-20T12:00:00Z"
}
```

### Error Response
```json
{
    "error": "Error message",
    "code": "ERROR_CODE",
    "details": "Additional error details",
    "timestamp": "2024-02-20T12:00:00Z"
}
```

## Best Practices

### 1. Error Handling
- Always check response status codes
- Implement proper error handling
- Log errors for debugging

### 2. Rate Limiting
- Implement client-side rate limiting
- Cache responses when possible
- Handle 429 responses gracefully

### 3. Performance
- Keep messages concise
- Implement request timeouts
- Use appropriate content types

## Future API Plans

### Planned Endpoints
1. User Management
   - User registration
   - Session management
   - Preference storage

2. Advanced Features
   - Multi-turn conversations
   - Custom personality settings
   - Conversation history

3. Analytics
   - Usage statistics
   - Performance metrics
   - Error tracking

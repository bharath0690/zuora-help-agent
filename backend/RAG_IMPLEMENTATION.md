# RAG Pipeline Implementation

Complete production-ready RAG (Retrieval-Augmented Generation) implementation in `/ask` endpoint.

## Overview

The `/ask` endpoint now implements a full RAG pipeline that:
1. Generates embeddings for user questions
2. Retrieves relevant documents from FAISS index
3. Generates contextual answers using LLM (Claude or OpenAI)
4. Returns structured responses with source citations

## Architecture

```
User Question
    ↓
Query Embedding (Voyage AI/OpenAI/Local)
    ↓
FAISS Vector Search (top-k similarity)
    ↓
Context Formatting (retrieved chunks)
    ↓
LLM Generation (Claude/GPT + System Prompt)
    ↓
Structured Response (answer + sources + confidence)
```

## Request Model

### AskRequest (Pydantic)

```python
{
  "question": str,           # Required, 1-1000 chars
  "conversation_id": str,    # Optional, for tracking
  "context": dict           # Optional, additional context
}
```

**Validation:**
- Minimum length: 1 character
- Maximum length: 1000 characters
- Automatic whitespace trimming
- Empty/whitespace-only questions rejected

**Example:**
```json
{
  "question": "What is Zuora CPQ?",
  "conversation_id": "conv_12345"
}
```

## Response Model

### AskResponse (Pydantic)

```python
{
  "answer": str,             # Generated answer
  "sources": List[Source],   # Source documents
  "conversation_id": str,    # Conversation ID
  "confidence": float        # 0.0-1.0
}
```

### Source Model

```python
{
  "title": str,              # Document title
  "source": str,             # Source file/URL
  "url": str,                # Optional: clickable URL
  "chunk_index": int,        # Optional: chunk position
  "similarity": float        # Optional: similarity score
}
```

**Example Response:**
```json
{
  "answer": "Zuora CPQ (Configure, Price, Quote) is a comprehensive solution...",
  "sources": [
    {
      "title": "Zuora CPQ Overview",
      "source": "zuora_cpq_guide.md",
      "url": "https://docs.zuora.com/cpq",
      "chunk_index": 0,
      "similarity": 0.89
    }
  ],
  "conversation_id": "conv_12345",
  "confidence": 0.8
}
```

## Pipeline Steps

### Step 1: Query Embedding

```python
query_embedding = await embedder.generate_query_embedding(request.question)
```

**Features:**
- Async embedding generation
- Provider-agnostic (Voyage AI/OpenAI/Local)
- Optimized for search queries (Voyage AI uses `input_type="query"`)
- Error handling with `EmbeddingError`

### Step 2: Document Retrieval

```python
results = index_loader.search(
    query_embedding=query_embedding,
    top_k=settings.TOP_K_RESULTS,      # Default: 5
    threshold=settings.SIMILARITY_THRESHOLD  # Default: 0.7
)
```

**Features:**
- FAISS L2 distance search
- Configurable top-k results
- Similarity threshold filtering
- Graceful fallback if index not available

**Retrieved Data:**
- Chunk text content
- Source metadata (title, URL)
- Similarity scores
- Chunk indices

### Step 3: Context Formatting

```python
user_prompt = create_rag_prompt(
    question=request.question,
    context_docs=relevant_docs,
)
```

**Prompt Structure:**
```
Based on the following documentation, please answer the user's question.

Documentation Context:
[Document 1 - Source: zuora_cpq_guide.md]
{chunk content}

[Document 2 - Source: zuora_billing_intro.md]
{chunk content}

...

User Question: {question}

Please provide a clear, accurate answer based on the documentation above.
```

### Step 4: LLM Generation

**Anthropic Claude:**
```python
response = llm_client.messages.create(
    model=settings.LLM_MODEL,          # e.g., claude-3-5-sonnet-20241022
    max_tokens=settings.LLM_MAX_TOKENS,  # Default: 1000
    temperature=settings.LLM_TEMPERATURE,  # Default: 0.7
    system=ZUORA_EXPERT_PROMPT,
    messages=[{"role": "user", "content": user_prompt}]
)
answer = response.content[0].text
```

**OpenAI:**
```python
response = llm_client.chat.completions.create(
    model=settings.LLM_MODEL,          # e.g., gpt-4-turbo-preview
    max_tokens=settings.LLM_MAX_TOKENS,
    temperature=settings.LLM_TEMPERATURE,
    messages=[
        {"role": "system", "content": ZUORA_EXPERT_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
)
answer = response.choices[0].message.content
```

### Step 5: Response Construction

```python
return AskResponse(
    answer=answer,
    sources=[...],                     # Formatted source objects
    conversation_id=request.conversation_id or generated_id,
    confidence=len(relevant_docs) / TOP_K_RESULTS  # Simple heuristic
)
```

## Error Handling

### HTTP Exceptions

| Code | Scenario | Example |
|------|----------|---------|
| **400** | Invalid question | Empty or whitespace-only |
| **503** | Service unavailable | LLM client not initialized |
| **500** | Processing error | Embedding/LLM generation failed |

### Graceful Degradation

1. **No FAISS Index:**
   - Logs warning
   - Proceeds with LLM knowledge only
   - Returns lower confidence

2. **Retrieval Failure:**
   - Logs error
   - Falls back to LLM knowledge
   - Response still generated

3. **Embedding Error:**
   - Returns 500 with specific error message
   - Request not processed

4. **LLM Error:**
   - Returns 500 with error details
   - Full error logging with stack trace

## Initialization (Startup)

### Lifespan Context Manager

Modern FastAPI pattern for startup/shutdown:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global embedder, index_loader, llm_client

    # Initialize embedder
    embedder = EmbeddingGenerator(...)

    # Initialize FAISS index (with fallback)
    try:
        index_loader = FAISSIndexLoader(...)
    except FileNotFoundError:
        index_loader = None  # Graceful fallback

    # Initialize LLM client
    if settings.LLM_PROVIDER == "anthropic":
        llm_client = anthropic.Anthropic(...)
    elif settings.LLM_PROVIDER == "openai":
        llm_client = openai.OpenAI(...)

    yield

    # Shutdown
    # Cleanup if needed
```

### Startup Logs

```
Starting Zuora Help Agent...
Environment: development
Initializing embedding generator...
✅ Embeddings: {'provider': 'voyage', 'model': 'voyage-2', 'dimension': 1024}
Loading FAISS index...
✅ FAISS Index: 150 chunks, 1024d
Initializing LLM client...
✅ LLM: Anthropic Claude (claude-3-5-sonnet-20241022)
🚀 Zuora Help Agent started successfully!
```

## Configuration

### Environment Variables

```bash
# Embedding Provider
EMBEDDING_PROVIDER=voyage              # voyage/openai/local
EMBEDDING_MODEL=voyage-2               # Provider-specific

# LLM Provider
LLM_PROVIDER=anthropic                 # anthropic/openai
LLM_MODEL=claude-3-5-sonnet-20241022  # Model name
LLM_TEMPERATURE=0.7                    # 0.0-1.0
LLM_MAX_TOKENS=1000                    # Max response length

# RAG Settings
TOP_K_RESULTS=5                        # Number of documents to retrieve
SIMILARITY_THRESHOLD=0.7               # Minimum similarity score

# API Keys
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
VOYAGE_API_KEY=pa-...
```

### Health Check

```bash
GET /health
```

Response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "environment": "development",
  "rag_enabled": true  // All components initialized
}
```

## Usage Examples

### Basic Question

```bash
curl -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "What is Zuora CPQ?"
  }'
```

### With Conversation ID

```bash
curl -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "How do I configure SSO?",
    "conversation_id": "conv_user_123"
  }'
```

### Error: Empty Question

```bash
curl -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": ""}'
```

Response:
```json
{
  "detail": "Question cannot be empty"
}
```

## Performance Metrics

Typical request flow:
1. **Embedding generation**: ~0.1-0.5s
2. **FAISS search**: ~0.01-0.05s
3. **LLM generation**: ~2-5s (depending on answer length)
4. **Total**: ~2-6s per request

## Logging

### Request Lifecycle

```
[INFO] Processing question: What is Zuora CPQ?...
[DEBUG] Generated query embedding: shape=(1024,)
[INFO] Retrieved 5 relevant documents
[INFO] Generated answer (847 chars)
```

### Error Logging

```
[ERROR] Embedding generation failed: No API key found
[ERROR] Document retrieval failed: FAISS index not loaded
[ERROR] LLM generation failed: Rate limit exceeded
[ERROR] Unexpected error processing question: ...
```

## Testing

### Test with Sample Data

```bash
# 1. Create sample index
cd scripts
python test_build_index.py

# 2. Build FAISS index
export VOYAGE_API_KEY="pa-..."
python build_index.py

# 3. Start backend
cd ../backend
export ANTHROPIC_API_KEY="sk-ant-..."
python main.py

# 4. Test endpoint
curl -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "What is Zuora CPQ?"}'
```

## Next Steps

### Enhancements

1. **Conversation History:**
   - Store conversation context
   - Support multi-turn conversations
   - Reference previous Q&A

2. **Caching:**
   - Cache embeddings for common questions
   - Cache LLM responses
   - Reduce API calls and costs

3. **Streaming:**
   - Stream LLM responses
   - Real-time answer generation
   - Better UX for long answers

4. **Advanced Retrieval:**
   - Hybrid search (vector + keyword)
   - Re-ranking retrieved documents
   - Query expansion

5. **Monitoring:**
   - Prometheus metrics
   - Request latency tracking
   - Error rate monitoring
   - Cost tracking (API calls)

## Troubleshooting

### "LLM service not initialized"

**Cause:** API key not set or invalid

**Fix:**
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
# or
export OPENAI_API_KEY="sk-..."
```

### "Embedding service not initialized"

**Cause:** Embedding provider configuration issue

**Fix:**
```bash
export VOYAGE_API_KEY="pa-..."
# or
export EMBEDDING_PROVIDER=openai
export OPENAI_API_KEY="sk-..."
```

### "Answer generation failed"

**Causes:**
- Rate limits exceeded
- Invalid API key
- Network issues

**Check logs** for specific error

### Empty Sources

**Cause:** FAISS index not built or not found

**Fix:**
```bash
cd scripts
python build_index.py --input ../data/processed_docs.json
```

## Security

- ✅ Input validation (Pydantic)
- ✅ Max question length (1000 chars)
- ✅ Error message sanitization
- ✅ No sensitive data in logs
- ✅ CORS configuration
- ✅ API key environment variables

## License

MIT License - For educational and development purposes.

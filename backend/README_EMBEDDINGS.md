# Embeddings Module Documentation

Production-ready embedding generation with multi-provider support, error handling, and retry logic.

## Important Note: Claude & Embeddings

**Claude (Anthropic) does not provide an embedding API.**

For RAG systems using Claude, Anthropic recommends using **Voyage AI** for embeddings. This module supports:
- ✅ **Voyage AI** (Anthropic's recommended partner)
- ✅ **OpenAI** (Alternative high-quality option)
- ✅ **Local models** (sentence-transformers, no API costs)

## Quick Start

### 1. Install Dependencies

```bash
pip install voyageai tiktoken numpy tenacity
# or
pip install -r requirements.txt
```

### 2. Set API Key

```bash
# Voyage AI (Anthropic recommended)
export VOYAGE_API_KEY="pa-..."

# OR OpenAI
export OPENAI_API_KEY="sk-..."
```

### 3. Basic Usage

```python
from embeddings import EmbeddingGenerator

# Initialize (auto-detects provider from env)
embedder = EmbeddingGenerator()

# Generate embeddings
texts = ["Zuora CPQ overview", "Subscription billing"]
embeddings = await embedder.generate_embeddings(texts)

# Shape: (2, 1024) for voyage-2
print(embeddings.shape)
```

## EmbeddingGenerator

### Initialization

```python
from embeddings import EmbeddingGenerator

# Default: Uses EMBEDDING_PROVIDER env var, falls back to "voyage"
embedder = EmbeddingGenerator()

# Explicit provider
embedder = EmbeddingGenerator(provider="openai")

# Custom model
embedder = EmbeddingGenerator(
    provider="voyage",
    model="voyage-large-2",
    batch_size=50,
    max_retries=5
)

# With API key override
embedder = EmbeddingGenerator(
    provider="openai",
    api_key="sk-...",
)
```

### Generate Embeddings

```python
# Multiple texts
texts = [
    "Zuora CPQ is a configure, price, quote solution.",
    "The Zuora platform enables subscription services.",
    "Payment processing supports multiple gateways.",
]

embeddings = await embedder.generate_embeddings(texts)
# Returns: np.ndarray of shape (3, dimension)

# Single query (optimized for search)
query = "How do I configure SSO?"
query_embedding = await embedder.generate_query_embedding(query)
# Returns: np.ndarray of shape (dimension,)
```

### Get Provider Info

```python
info = embedder.get_provider_info()
# {
#     "provider": "voyage",
#     "model": "voyage-2",
#     "dimension": 1024,
#     "batch_size": 100
# }

dimension = embedder.get_dimension()  # e.g., 1024
```

## Providers

### Voyage AI (Anthropic Recommended)

**Best for:** RAG systems using Claude

**Setup:**
```bash
export VOYAGE_API_KEY="pa-..."
```

**Models:**
- `voyage-2` (1024 dims) - General purpose, default
- `voyage-large-2` (1536 dims) - Higher capacity
- `voyage-code-2` (1536 dims) - Optimized for code

**Features:**
- Optimized for retrieval tasks
- Separate "document" and "query" input types
- Recommended by Anthropic for Claude RAG

**Usage:**
```python
embedder = EmbeddingGenerator(provider="voyage", model="voyage-2")

# Document embeddings
doc_embeddings = await embedder.generate_embeddings(docs)

# Query embeddings (optimized)
query_embedding = await embedder.generate_query_embedding(query)
```

**Pricing:** ~$0.10 per 1M tokens

### OpenAI

**Best for:** High-quality embeddings, well-documented

**Setup:**
```bash
export OPENAI_API_KEY="sk-..."
```

**Models:**
- `text-embedding-3-small` (1536 dims) - Fast, cheap, default
- `text-embedding-3-large` (3072 dims) - Higher quality
- `text-embedding-ada-002` (1536 dims) - Legacy

**Usage:**
```python
embedder = EmbeddingGenerator(
    provider="openai",
    model="text-embedding-3-small"
)

embeddings = await embedder.generate_embeddings(texts)
```

**Pricing:**
- `text-embedding-3-small`: $0.02 per 1M tokens
- `text-embedding-3-large`: $0.13 per 1M tokens

### Local (sentence-transformers)

**Best for:** Privacy, no API costs, offline use

**Setup:**
```bash
pip install sentence-transformers
# No API key needed
```

**Models:**
- `all-MiniLM-L6-v2` (384 dims) - Fast, small, default
- `all-mpnet-base-v2` (768 dims) - Better quality
- `all-MiniLM-L12-v2` (384 dims) - Balance

**Usage:**
```python
embedder = EmbeddingGenerator(
    provider="local",
    model="all-MiniLM-L6-v2"
)

embeddings = await embedder.generate_embeddings(texts)
```

**Pros:**
- No API costs
- No rate limits
- Privacy (data stays local)

**Cons:**
- Slower without GPU
- Lower quality for domain-specific content
- Requires model download (~100MB)

## DocumentChunker

### Initialization

```python
from embeddings import DocumentChunker

# Default settings from config
chunker = DocumentChunker()

# Custom settings
chunker = DocumentChunker(
    chunk_size=800,      # tokens
    chunk_overlap=100,   # tokens
    chunking_strategy="recursive"
)
```

### Chunk Text

```python
document = """
Long document text here...
Multiple paragraphs...
"""

metadata = {
    "source": "zuora_guide.md",
    "product": "billing",
    "url": "https://docs.zuora.com/..."
}

chunks = chunker.chunk_text(document, metadata)

# Returns list of dicts:
# [
#     {
#         "content": "Chunk text...",
#         "metadata": {
#             "source": "zuora_guide.md",
#             "product": "billing",
#             "url": "https://...",
#             "chunk_index": 0,
#             "token_count": 756,
#             "chunked_at": "2024-02-16T10:30:00"
#         }
#     },
#     ...
# ]
```

### Token Counting

```python
# Accurate with tiktoken (recommended)
token_count = chunker.count_tokens("Some text")

# Falls back to approximation if tiktoken not available
# ~4 characters per token
```

## Error Handling

### EmbeddingError

Custom exception for embedding-related failures:

```python
from embeddings import EmbeddingError

try:
    embeddings = await embedder.generate_embeddings(texts)
except EmbeddingError as e:
    print(f"Embedding failed: {e}")
    # Handle error (log, retry, fallback, etc.)
```

### Retry Logic

Built-in retry with exponential backoff:

```python
# Automatically retries up to 3 times
# Wait times: 2s, 4s, 8s (exponential)
# Only retries on transient errors (network, rate limits)

embedder = EmbeddingGenerator(max_retries=5)  # Custom retry count
```

### Batch Failure Handling

```python
# If a batch fails, it's retried once
# If retry fails, fills with zero vectors
# Raises error if >10% of batches fail

# Check for zero vectors in results
if (embeddings == 0).all(axis=1).any():
    print("Warning: Some embeddings failed")
```

## Batching

### Default Batching

```python
embedder = EmbeddingGenerator(batch_size=100)

# Processes 250 texts in 3 batches (100, 100, 50)
texts = ["text"] * 250
embeddings = await embedder.generate_embeddings(texts)
```

### Custom Batch Size

```python
# Smaller batches (safer, slower)
embedder = EmbeddingGenerator(batch_size=50)

# Larger batches (faster, may hit rate limits)
embedder = EmbeddingGenerator(batch_size=500)
```

### Rate Limiting

Built-in 0.1s delay between batches for API providers:

```python
# Automatic rate limiting
# 100 texts/batch = max ~1000 texts/sec
```

## Complete Example

```python
import asyncio
from embeddings import EmbeddingGenerator, DocumentChunker

async def process_document():
    # Load document
    with open("zuora_guide.md") as f:
        document = f.read()

    # Chunk document
    chunker = DocumentChunker(chunk_size=800, chunk_overlap=100)
    chunks = chunker.chunk_text(document, metadata={
        "source": "zuora_guide.md",
        "product": "billing"
    })

    print(f"Created {len(chunks)} chunks")

    # Generate embeddings
    embedder = EmbeddingGenerator(provider="voyage")

    chunk_texts = [chunk["content"] for chunk in chunks]
    embeddings = await embedder.generate_embeddings(chunk_texts)

    print(f"Generated {len(embeddings)} embeddings")
    print(f"Dimension: {embedder.get_dimension()}")

    # Add embeddings to chunks
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding

    # Store in vector database
    # ... (your vector store code)

asyncio.run(process_document())
```

## Testing

Run the test suite:

```bash
# Set API key
export VOYAGE_API_KEY="pa-..."

# Run tests
cd backend
python test_embeddings.py
```

Tests cover:
1. ✅ Embedding generation
2. ✅ Query embeddings
3. ✅ Batch processing
4. ✅ Document chunking
5. ✅ Error handling
6. ✅ Local embeddings (optional)

## Environment Variables

```bash
# Required (choose one)
VOYAGE_API_KEY=pa-...           # Voyage AI
OPENAI_API_KEY=sk-...           # OpenAI

# Optional
EMBEDDING_PROVIDER=voyage       # voyage/openai/local
EMBEDDING_MODEL=voyage-2        # Provider-specific model
EMBEDDING_BATCH_SIZE=100        # Batch size
```

## Best Practices

### 1. Choose the Right Provider

**Use Voyage AI if:**
- Building RAG with Claude
- Want Anthropic-recommended solution
- Need optimized retrieval embeddings

**Use OpenAI if:**
- Want well-documented API
- Need higher dimensions (3072d)
- Already using OpenAI ecosystem

**Use Local if:**
- Privacy is critical
- Want no API costs
- Have GPU available

### 2. Optimize Batch Size

```python
# For API providers with rate limits
embedder = EmbeddingGenerator(batch_size=50)

# For local models with GPU
embedder = EmbeddingGenerator(batch_size=500)
```

### 3. Handle Errors Gracefully

```python
try:
    embeddings = await embedder.generate_embeddings(texts)
except EmbeddingError as e:
    logger.error(f"Embedding failed: {e}")
    # Fallback to cached embeddings or retry later
```

### 4. Use Appropriate Chunk Sizes

```python
# For Claude (200K context)
chunker = DocumentChunker(chunk_size=800, chunk_overlap=100)

# For GPT-4 (128K context)
chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)
```

### 5. Monitor Performance

```python
import time

start = time.time()
embeddings = await embedder.generate_embeddings(texts)
elapsed = time.time() - start

print(f"Generated {len(texts)} embeddings in {elapsed:.2f}s")
print(f"Rate: {len(texts)/elapsed:.1f} texts/sec")
```

## Troubleshooting

### "No API key found"

```bash
# Set the appropriate API key
export VOYAGE_API_KEY="pa-..."
# or
export OPENAI_API_KEY="sk-..."
```

### "ImportError: voyageai"

```bash
pip install voyageai
```

### "Rate limit exceeded"

```python
# Reduce batch size
embedder = EmbeddingGenerator(batch_size=25)

# Or increase retry delay
# (automatic exponential backoff: 2s, 4s, 8s)
```

### "Too many failed batches"

```python
# Check API key is valid
# Check network connection
# Check API service status

# Reduce batch size
embedder = EmbeddingGenerator(batch_size=10)
```

### "tiktoken not found"

```bash
pip install tiktoken
# Falls back to approximate counting if unavailable
```

## Performance

### Benchmarks

**Voyage AI (voyage-2):**
- ~1000 texts/sec (batch_size=100)
- ~0.1s latency per batch

**OpenAI (text-embedding-3-small):**
- ~500 texts/sec (batch_size=100)
- ~0.2s latency per batch

**Local (all-MiniLM-L6-v2):**
- ~100 texts/sec (CPU)
- ~2000 texts/sec (GPU)

### Cost Comparison

For 1M tokens (≈750K words):

- **Voyage AI**: ~$0.10
- **OpenAI small**: ~$0.02
- **OpenAI large**: ~$0.13
- **Local**: $0 (free)

## Integration with RAG Pipeline

```python
from embeddings import EmbeddingGenerator
from rag import DocumentRetriever, RAGPipeline

# Initialize embedder
embedder = EmbeddingGenerator(provider="voyage")

# Generate query embedding
query = "How do I configure SSO?"
query_embedding = await embedder.generate_query_embedding(query)

# Retrieve similar documents
retriever = DocumentRetriever(vector_store_path="../data/vector_store")
docs = await retriever.retrieve(query_embedding, top_k=5)

# Generate answer with Claude
pipeline = RAGPipeline()
answer = await pipeline.generate_answer(query, docs)
```

## API Reference

### Classes

- **EmbeddingGenerator**: Generate embeddings with multi-provider support
- **DocumentChunker**: Split documents into token-sized chunks
- **EmbeddingError**: Custom exception for embedding errors

### Methods

**EmbeddingGenerator:**
- `__init__(provider, model, api_key, batch_size, max_retries)`
- `generate_embeddings(texts, show_progress) -> np.ndarray`
- `generate_query_embedding(query) -> np.ndarray`
- `get_dimension() -> int`
- `get_provider_info() -> Dict`

**DocumentChunker:**
- `__init__(chunk_size, chunk_overlap, chunking_strategy)`
- `chunk_text(text, metadata) -> List[Dict]`
- `count_tokens(text) -> int`

## License

MIT License - For educational and development purposes.

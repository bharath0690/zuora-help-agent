# Document Ingestion Guide

Complete guide for chunking documentation, generating embeddings, and building the FAISS vector store.

## Overview

The ingestion pipeline:
1. **Loads** scraped JSON documentation
2. **Chunks** content into 800-token sections with overlap
3. **Generates** embeddings using OpenAI or Voyage AI
4. **Stores** vectors in FAISS for fast similarity search
5. **Saves** metadata for retrieval

## Quick Start

### 1. Install Dependencies

```bash
cd scripts
pip install -r requirements-ingestion.txt
```

### 2. Set API Key

For OpenAI (default):
```bash
export OPENAI_API_KEY="sk-..."
```

For Voyage AI (Anthropic's recommended partner):
```bash
export VOYAGE_API_KEY="pa-..."
```

### 3. Run Ingestion

```bash
# Using OpenAI embeddings (default)
python ingest_docs.py --input ../data/zuora_docs.json

# Using Voyage AI embeddings (recommended for Claude RAG)
python ingest_docs.py --provider voyage --input ../data/zuora_docs.json
```

### 4. Test the Index

```bash
python query_index.py --query "What is Zuora CPQ?"
```

## Detailed Usage

### Ingestion Script

**Basic Usage:**
```bash
python ingest_docs.py --input ../data/zuora_docs.json
```

**All Options:**
```bash
python ingest_docs.py \
    --input ../data/zuora_docs.json \
    --output ../data/vector_store \
    --provider openai \
    --model text-embedding-3-small \
    --chunk-size 800 \
    --chunk-overlap 100 \
    --index-type flat \
    --batch-size 100
```

**Command-Line Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | `../data/zuora_docs.json` | Input JSON file with scraped docs |
| `--output` | `../data/vector_store` | Output directory for FAISS index |
| `--provider` | `openai` | Embedding provider (openai/voyage/local) |
| `--model` | Auto | Embedding model name |
| `--chunk-size` | `800` | Chunk size in tokens |
| `--chunk-overlap` | `100` | Overlap between chunks in tokens |
| `--index-type` | `flat` | FAISS index type (flat/ivf) |
| `--batch-size` | `100` | Batch size for embedding API |

### Query Script

**Basic Query:**
```bash
python query_index.py --query "What is Zuora CPQ?"
```

**More Results:**
```bash
python query_index.py --query "How to configure SSO?" --top-k 10
```

**Custom Index Path:**
```bash
python query_index.py \
    --index ../data/vector_store_v2 \
    --query "Payment methods" \
    --top-k 5
```

## Embedding Providers

### OpenAI (Default)

**Pros:**
- High quality embeddings
- Well-documented API
- Fast and reliable

**Models:**
- `text-embedding-3-small` (1536 dims, $0.02/1M tokens) - Default
- `text-embedding-3-large` (3072 dims, $0.13/1M tokens) - Higher quality
- `text-embedding-ada-002` (1536 dims, $0.10/1M tokens) - Legacy

**Setup:**
```bash
export OPENAI_API_KEY="sk-..."
python ingest_docs.py --provider openai --model text-embedding-3-small
```

### Voyage AI (Anthropic Recommended)

**Pros:**
- Recommended by Anthropic for Claude RAG
- Optimized for retrieval tasks
- Competitive pricing

**Models:**
- `voyage-2` (1024 dims) - General purpose
- `voyage-large-2` (1536 dims) - Higher capacity
- `voyage-code-2` (1536 dims) - For code

**Setup:**
```bash
pip install voyageai
export VOYAGE_API_KEY="pa-..."
python ingest_docs.py --provider voyage --model voyage-2
```

### Local Embeddings

**Pros:**
- No API costs
- No API rate limits
- Privacy (data stays local)

**Cons:**
- Slower
- Lower quality for domain-specific content
- Requires GPU for good performance

**Models:**
- `all-MiniLM-L6-v2` (384 dims) - Fast, small
- `all-mpnet-base-v2` (768 dims) - Better quality
- `all-MiniLM-L12-v2` (384 dims) - Balance

**Setup:**
```bash
pip install sentence-transformers
python ingest_docs.py --provider local --model all-MiniLM-L6-v2
```

## Chunking Strategy

### Token-Based Chunking

Documents are split into chunks based on token count (not character count):

**Default Settings:**
- **Chunk Size**: 800 tokens (~3200 characters)
- **Overlap**: 100 tokens (~400 characters)

**Why 800 tokens?**
- Fits well within Claude's context window
- Captures complete concepts (paragraphs, code blocks)
- Balances precision vs. context
- Efficient for retrieval

**Overlap Benefits:**
- Prevents context loss at boundaries
- Improves retrieval for concepts spanning chunks
- 100 tokens = ~1-2 sentences of overlap

### Chunking Algorithm

1. **Split by paragraphs** (`\n\n`)
2. **Count tokens** using tiktoken
3. **Combine paragraphs** until chunk_size reached
4. **Split long paragraphs** by sentences if needed
5. **Add overlap** from previous chunk
6. **Preserve structure** (headings, code blocks)

### Custom Chunk Sizes

```bash
# Smaller chunks (more precise, less context)
python ingest_docs.py --chunk-size 500 --chunk-overlap 50

# Larger chunks (more context, less precise)
python ingest_docs.py --chunk-size 1200 --chunk-overlap 200

# No overlap (faster, but may lose boundary context)
python ingest_docs.py --chunk-size 800 --chunk-overlap 0
```

## FAISS Index Types

### Flat Index (Default)

**Type:** `IndexFlatL2`

**Characteristics:**
- Exact similarity search (100% recall)
- Fast for < 1M vectors
- No training required
- Higher memory usage

**Best for:**
- Small to medium datasets (< 100K chunks)
- When accuracy is critical
- Development and testing

**Usage:**
```bash
python ingest_docs.py --index-type flat
```

### IVF Index

**Type:** `IndexIVFFlat`

**Characteristics:**
- Approximate search (~99% recall)
- Faster for large datasets
- Requires training
- Lower memory with quantization

**Best for:**
- Large datasets (> 100K chunks)
- Production deployments
- When speed > perfect accuracy

**Usage:**
```bash
python ingest_docs.py --index-type ivf
```

## Output Structure

After ingestion, the output directory contains:

```
vector_store/
├── faiss.index              # FAISS index binary
├── chunks_metadata.pkl      # Chunk metadata (pickled)
├── index_config.json        # Index configuration
└── ingestion_summary.txt    # Human-readable summary
```

### faiss.index

Binary FAISS index containing all embeddings.

### chunks_metadata.pkl

Pickled list of chunk metadata:
```python
[
    {
        "chunk_id": "doc_0_chunk_0",
        "content": "Zuora CPQ is a configure, price, quote...",
        "metadata": {
            "url": "https://docs.zuora.com/...",
            "title": "Zuora CPQ Overview",
            "product": "cpq",
            "chunk_index": 0,
            "doc_index": 0,
            "word_count": 245,
            "chunked_at": "2024-02-16T10:30:00"
        },
        "token_count": 187
    },
    # ... more chunks
]
```

### index_config.json

Index metadata:
```json
{
  "dimension": 1536,
  "index_type": "flat",
  "num_chunks": 1250,
  "created_at": "2024-02-16T10:30:00"
}
```

### ingestion_summary.txt

Human-readable summary:
```
Document Ingestion Summary
============================================================

Source: ../data/zuora_docs.json
Documents: 150
Chunks: 1250
Chunk Size: 800 tokens
Chunk Overlap: 100 tokens
Embedding Provider: openai
Embedding Model: text-embedding-3-small
Embedding Dimension: 1536
Index Type: flat
Created: 2024-02-16T10:30:00

Token Statistics:
  Min: 245
  Max: 895
  Mean: 756.3
  Median: 782.0
```

## Cost Estimation

### OpenAI Embeddings

**text-embedding-3-small** ($0.02 per 1M tokens):
- 1,000 chunks × 800 tokens = 800,000 tokens
- Cost: ~$0.016

**text-embedding-3-large** ($0.13 per 1M tokens):
- 1,000 chunks × 800 tokens = 800,000 tokens
- Cost: ~$0.10

### Voyage AI Embeddings

**voyage-2** (~$0.10 per 1M tokens):
- Similar to OpenAI ada-002
- 1,000 chunks × 800 tokens ≈ $0.08

### Example: Full Zuora Docs

Assuming 500 pages scraped:
- Average 1,000 words/page
- ~1,250 tokens/page
- Total: 625,000 tokens
- Chunks: ~800-900 chunks
- **OpenAI Cost**: ~$0.01-$0.02
- **Voyage Cost**: ~$0.06-$0.08

## Performance Optimization

### Batch Size

Adjust based on API rate limits:

```bash
# Larger batches (faster, but may hit rate limits)
python ingest_docs.py --batch-size 500

# Smaller batches (slower, but safer)
python ingest_docs.py --batch-size 50
```

### GPU Acceleration

For FAISS with GPU:

```bash
# Install GPU version
pip uninstall faiss-cpu
pip install faiss-gpu

# FAISS will auto-detect and use GPU
python ingest_docs.py
```

### Parallel Processing

For very large datasets, process in chunks:

```bash
# Process first 100 docs
python ingest_docs.py --input zuora_docs_part1.json --output store_part1

# Process next 100 docs
python ingest_docs.py --input zuora_docs_part2.json --output store_part2

# Merge indices (advanced - requires custom script)
```

## Troubleshooting

### "tiktoken not installed"

```bash
pip install tiktoken
```

### "No API key found"

```bash
export OPENAI_API_KEY="sk-..."
# or
export VOYAGE_API_KEY="pa-..."
```

### "Rate limit exceeded"

Reduce batch size:
```bash
python ingest_docs.py --batch-size 50
```

### "Out of memory"

Use IVF index or process in smaller batches:
```bash
python ingest_docs.py --index-type ivf --batch-size 50
```

### "Import Error: openai"

```bash
pip install openai
```

### Query returns no results

Ensure you're using the same embedding provider/model:
```bash
# Ingestion
python ingest_docs.py --provider openai --model text-embedding-3-small

# Query (must match)
python query_index.py --provider openai --model text-embedding-3-small --query "..."
```

## Next Steps

After ingestion:

1. **Test retrieval**: Use `query_index.py` to verify search quality
2. **Integrate with RAG**: Update `backend/rag.py` to use FAISS index
3. **Deploy**: Copy vector store to production environment
4. **Monitor**: Track retrieval quality and update as docs change

## Integration with Backend

Update `backend/rag.py` to use the FAISS index:

```python
import faiss
import pickle
from pathlib import Path

class DocumentRetriever:
    def __init__(self, vector_store_path: str):
        # Load index
        index_path = Path(vector_store_path) / "faiss.index"
        self.index = faiss.read_index(str(index_path))

        # Load metadata
        metadata_path = Path(vector_store_path) / "chunks_metadata.pkl"
        with open(metadata_path, "rb") as f:
            self.chunks_metadata = pickle.load(f)

    async def retrieve(self, query_embedding: np.ndarray, top_k: int = 5):
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx in indices[0]:
            results.append(self.chunks_metadata[idx])

        return results
```

## Best Practices

1. **Version your indices**: Use dated output directories
   ```bash
   python ingest_docs.py --output ../data/vector_store_2024_02_16
   ```

2. **Keep source data**: Don't delete `zuora_docs.json` - you may need to re-ingest

3. **Test before deploying**: Always run `query_index.py` to verify quality

4. **Monitor costs**: Track embedding API usage

5. **Update regularly**: Re-scrape and re-ingest as docs change

6. **Document your config**: Save ingestion commands in a script

## License

MIT License - For educational and development purposes.

# Zuora Help Agent

A production-ready RAG (Retrieval-Augmented Generation) based AI assistant for Zuora product documentation and support. Built with FastAPI, LangChain, and vector databases.

## Features

- 🤖 **Intelligent Q&A**: Ask questions about Zuora products and get accurate answers
- 📚 **RAG Pipeline**: Retrieval-Augmented Generation for contextual responses
- 🔍 **Semantic Search**: Vector-based document retrieval
- 💬 **Conversation Tracking**: Maintains context across multiple questions
- 🚀 **Production Ready**: Health checks, logging, error handling
- 🔌 **Flexible LLM Support**: OpenAI and Anthropic (Claude)
- 📦 **Multiple Vector Stores**: ChromaDB, Pinecone, Weaviate support

## Project Structure

```
zuora-help-agent/
├── backend/
│   ├── main.py              # FastAPI application & endpoints
│   ├── config.py            # Configuration & environment variables
│   ├── rag.py               # RAG pipeline implementation
│   ├── embeddings.py        # Embedding generation & vector store
│   ├── faiss_loader.py      # Runtime FAISS index loader
│   └── prompts.py           # Prompt templates & utilities
├── frontend/
│   ├── index.html           # Chat interface
│   ├── style.css            # Styles
│   ├── app.js               # JavaScript logic
│   ├── serve.py             # Development server
│   └── README.md            # Frontend documentation
├── data/
│   ├── zuora_docs.json      # Scraped documentation
│   ├── processed_docs.json  # Processed documents
│   ├── faiss.index          # FAISS vector index
│   ├── metadata.json        # Index metadata
│   └── vector_store/        # Alternative vector store
├── scripts/
│   ├── scrape_zuora_sitemap.py    # Sitemap-based scraper
│   ├── scrape_zuora_docs.py       # Web crawler scraper
│   ├── ingest_docs.py             # Chunking + embedding + FAISS
│   ├── build_index.py             # Build runtime FAISS index
│   ├── query_index.py             # Test vector store queries
│   ├── requirements-scraper.txt   # Scraping dependencies
│   ├── requirements-ingestion.txt # Ingestion dependencies
│   ├── README_SCRAPING.md         # Scraping guide
│   └── README_INGESTION.md        # Ingestion guide
├── requirements.txt         # Core backend dependencies
├── .env.example            # Example environment variables
└── README.md               # This file
```

## Prerequisites

- Python 3.11+
- OpenAI API key or Anthropic API key
- (Optional) Pinecone account for cloud vector storage

## Setup

### 1. Clone and Navigate

```bash
cd zuora-help-agent
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```env
OPENAI_API_KEY=sk-...your-key-here...
# OR
ANTHROPIC_API_KEY=sk-ant-...your-key-here...
```

### 5. Run the Application

```bash
cd backend
python main.py
```

Or using uvicorn directly:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

### Health Check

```bash
GET /health
```

Response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "environment": "development"
}
```

### Ask Question

```bash
POST /ask
Content-Type: application/json

{
  "question": "What is Zuora CPQ?",
  "conversation_id": "optional_conversation_id",
  "context": {}
}
```

Response:
```json
{
  "answer": "Zuora CPQ (Configure, Price, Quote) is...",
  "sources": ["zuora_cpq_guide.pdf", "api_docs.md"],
  "conversation_id": "conv_12345",
  "confidence": 0.92
}
```

## API Documentation

Interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Frontend

A minimal, clean chat interface is available in `frontend/`:

### Quick Start

```bash
# Start backend first
cd backend
python main.py

# In another terminal, start frontend
cd frontend
python serve.py

# Opens browser at http://localhost:3000
```

### Features

- ✅ Simple HTML + CSS + JavaScript (no frameworks)
- ✅ Clean, modern chat interface
- ✅ Connects to `/ask` endpoint
- ✅ Shows answers with clickable sources
- ✅ Typing indicators and loading states
- ✅ Error handling
- ✅ Fully responsive (desktop, tablet, mobile)

See [frontend/README.md](frontend/README.md) for detailed documentation.

## Configuration Options

### LLM Providers

**OpenAI (default)**:
```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-4-turbo-preview
OPENAI_API_KEY=your_key
```

**Anthropic Claude**:
```env
LLM_PROVIDER=anthropic
LLM_MODEL=claude-3-opus-20240229
ANTHROPIC_API_KEY=your_key
```

### Vector Stores

**ChromaDB (default, local)**:
```env
VECTOR_STORE_TYPE=chroma
CHROMA_PERSIST_DIRECTORY=../data/chroma_db
```

**Pinecone (cloud)**:
```env
VECTOR_STORE_TYPE=pinecone
PINECONE_API_KEY=your_key
PINECONE_ENVIRONMENT=us-east-1
PINECONE_INDEX_NAME=zuora-help
```

### RAG Settings

```env
CHUNK_SIZE=1000           # Document chunk size
CHUNK_OVERLAP=200         # Overlap between chunks
TOP_K_RESULTS=5           # Number of documents to retrieve
SIMILARITY_THRESHOLD=0.7  # Minimum similarity score
```

## Development

### Code Quality

Format code:
```bash
black backend/
```

Lint code:
```bash
ruff backend/
```

Type checking:
```bash
mypy backend/
```

### Running Tests

```bash
pytest
```

With coverage:
```bash
pytest --cov=backend --cov-report=html
```

## Data Collection & Ingestion

### Step 1: Scrape Zuora Documentation

Two scraping tools are available in `scripts/`:

**Option 1: Sitemap-based Scraper (Recommended)**

Efficient scraping using Zuora's sitemap.xml:

```bash
cd scripts

# Install scraping dependencies
pip install -r requirements-scraper.txt

# Scrape specific product (billing)
python scrape_zuora_sitemap.py --product billing --max-pages 100

# Scrape multiple products
python scrape_zuora_sitemap.py --product billing payments platform --max-pages 300

# Scrape all Zuora products
python scrape_zuora_sitemap.py --all-products --max-pages 1000
```

Available products: `billing`, `payments`, `platform`, `cpq`, `revenue`, `ar`, `zephr`, `basics`, `entitlements`, `release-notes`

**Option 2: Web Crawler**

General-purpose crawler for docs.zuora.com:

```bash
python scrape_zuora_docs.py --max-pages 200 --delay 1.5
```

Both scrapers output to `../data/zuora_docs.json` with cleaned text and metadata.

See [scripts/README_SCRAPING.md](scripts/README_SCRAPING.md) for detailed documentation.

### Step 2: Generate Embeddings & Build Vector Store

After scraping, process the documentation into a searchable FAISS index:

```bash
# Install ingestion dependencies
pip install -r requirements-ingestion.txt

# Set API key (OpenAI or Voyage AI)
export OPENAI_API_KEY="sk-..."

# Run ingestion (chunks docs, generates embeddings, builds FAISS index)
python ingest_docs.py --input ../data/zuora_docs.json
```

**Key Features:**
- ✅ Chunks documents into 800-token sections with 100-token overlap
- ✅ Generates embeddings using OpenAI or Voyage AI (Anthropic recommended)
- ✅ Stores vectors in FAISS for fast similarity search
- ✅ Preserves metadata for source attribution

**Test the Index:**
```bash
python query_index.py --query "What is Zuora CPQ?"
```

See [scripts/README_INGESTION.md](scripts/README_INGESTION.md) for detailed documentation.

## Next Steps

### 1. Collect & Process Documentation ✅

- [x] Scrape Zuora documentation (use `scrape_zuora_sitemap.py`)
- [x] Chunk and embed documents (use `ingest_docs.py`)
- [x] Build FAISS vector store

### 2. Integrate RAG Pipeline

Update `backend/rag.py` to use the FAISS index:

```python
import faiss
import pickle
from pathlib import Path
import numpy as np

class DocumentRetriever:
    def __init__(self, vector_store_path: str = "../data/vector_store"):
        # Load FAISS index
        index_path = Path(vector_store_path) / "faiss.index"
        self.index = faiss.read_index(str(index_path))

        # Load chunk metadata
        metadata_path = Path(vector_store_path) / "chunks_metadata.pkl"
        with open(metadata_path, "rb") as f:
            self.chunks = pickle.load(f)

    async def retrieve(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[Dict]:
        """Retrieve most similar chunks."""
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx, distance in zip(indices[0], distances[0]):
            chunk = self.chunks[idx]
            chunk["similarity_score"] = float(distance)
            results.append(chunk)

        return results
```

### 3. Complete Answer Generation

Implement the LLM integration in `rag.py`:

```python
from anthropic import Anthropic

class AnswerGenerator:
    def __init__(self):
        self.client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)

    async def generate(
        self,
        query: str,
        context_chunks: List[Dict],
        conversation_history: List[Dict] = None
    ) -> str:
        """Generate answer using Claude with retrieved context."""
        # Format context from chunks
        context = "\n\n".join([
            f"[Source: {c['metadata']['title']}]\n{c['content']}"
            for c in context_chunks
        ])

        # Build prompt
        system_prompt = get_system_prompt()
        user_prompt = format_rag_prompt(query, context)

        # Call Claude
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2000,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )

        return response.content[0].text
```

### 4. Add Conversation Storage

Implement conversation history tracking:
- Use Redis for short-term cache
- Use PostgreSQL for long-term storage

### 5. Enhance Monitoring

Add:
- Prometheus metrics
- Request logging
- Performance tracking
- Error alerting

### 6. Deploy

Options:
- **Docker**: Create Dockerfile and docker-compose.yml
- **Cloud**: Deploy to AWS, GCP, or Azure
- **Serverless**: Use AWS Lambda or Google Cloud Functions

## Environment Variables Reference

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | - |
| `ANTHROPIC_API_KEY` | Anthropic API key | - |
| `VECTOR_STORE_TYPE` | Vector store (chroma/pinecone) | chroma |
| `LLM_PROVIDER` | LLM provider (openai/anthropic) | openai |
| `LLM_MODEL` | Model name | gpt-4-turbo-preview |
| `CHUNK_SIZE` | Document chunk size | 1000 |
| `TOP_K_RESULTS` | Retrieval result count | 5 |

See `.env.example` for complete list.

## Troubleshooting

### Import Errors

Make sure you're running from the `backend/` directory:
```bash
cd backend
python main.py
```

### API Key Issues

Verify your API keys are set:
```bash
echo $OPENAI_API_KEY
```

### Port Already in Use

Change the port in `.env`:
```env
PORT=8001
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run code quality checks
5. Submit pull request

## License

MIT License - see LICENSE file for details.

## Support

For issues and questions:
- GitHub Issues: [Create an issue](https://github.com/your-repo/issues)
- Documentation: See `/docs` endpoint
- Email: help@bharathmarimuthu.in

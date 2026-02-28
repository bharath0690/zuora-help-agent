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
│   ├── main.py           # FastAPI application & endpoints
│   ├── config.py         # Configuration & environment variables
│   ├── rag.py            # RAG pipeline implementation
│   ├── embeddings.py     # Embedding generation & vector store
│   └── prompts.py        # Prompt templates & utilities
├── data/                 # Data storage (vector DB, documents)
├── scripts/              # Utility scripts (ingestion, etc.)
├── requirements.txt      # Python dependencies
├── .env.example         # Example environment variables
└── README.md            # This file
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

## Data Collection

### Scraping Zuora Documentation

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

## Next Steps

### 1. Collect Documentation

Use the scraping tools above to gather Zuora documentation.

### 2. Implement RAG Components

- [ ] Complete `embeddings.py` - Vector store integration
- [ ] Complete `rag.py` - Document retrieval and answer generation
- [ ] Create document ingestion pipeline

### 3. Process Documents

Create script to ingest scraped documentation:

```python
# scripts/ingest_docs.py
from pathlib import Path
from backend.embeddings import DocumentProcessor
import json

processor = DocumentProcessor()

# Load scraped docs
with open("../data/zuora_docs.json") as f:
    data = json.load(f)

# Process each document
for doc in data["documents"]:
    await processor.process_document(
        content=doc["content"],
        metadata=doc["metadata"]
    )
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

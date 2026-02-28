#!/bin/bash
#
# Zuora Help Agent - Complete Demo Workflow
#
# This script demonstrates the complete pipeline:
# 1. Scrape Zuora documentation
# 2. Generate embeddings and build FAISS index
# 3. Query the vector store
#
# Usage:
#   chmod +x demo.sh
#   ./demo.sh

set -e  # Exit on error

echo "=========================================="
echo "Zuora Help Agent - Complete Demo"
echo "=========================================="
echo ""

# Check for API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ Error: OPENAI_API_KEY not set"
    echo ""
    echo "Please set your API key:"
    echo "  export OPENAI_API_KEY='sk-...'"
    echo ""
    exit 1
fi

echo "✅ API key found"
echo ""

# Step 1: Install dependencies
echo "📦 Step 1: Installing dependencies..."
echo "=========================================="

if ! pip show requests &> /dev/null; then
    echo "Installing scraping dependencies..."
    pip install -q -r requirements-scraper.txt
else
    echo "Scraping dependencies already installed"
fi

if ! pip show faiss-cpu &> /dev/null; then
    echo "Installing ingestion dependencies..."
    pip install -q -r requirements-ingestion.txt
else
    echo "Ingestion dependencies already installed"
fi

echo "✅ Dependencies installed"
echo ""

# Step 2: Scrape documentation
echo "🕷️  Step 2: Scraping Zuora documentation..."
echo "=========================================="
echo "This will scrape up to 50 pages from Zuora Billing docs"
echo ""

if [ ! -f "../data/zuora_docs.json" ]; then
    python scrape_zuora_sitemap.py \
        --product billing \
        --max-pages 50 \
        --delay 0.5 \
        --output ../data/zuora_docs.json
else
    echo "ℹ️  Scraped data already exists at ../data/zuora_docs.json"
    echo "   Delete it to re-scrape, or press Enter to continue..."
    read
fi

echo ""
echo "✅ Scraping complete"
echo ""

# Check if docs were scraped
if [ ! -f "../data/zuora_docs.json" ]; then
    echo "❌ Error: Scraping failed - zuora_docs.json not found"
    exit 1
fi

# Show summary
echo "📊 Scraped data summary:"
python -c "
import json
with open('../data/zuora_docs.json') as f:
    data = json.load(f)
    print(f\"   Documents: {len(data.get('documents', []))}\")
    print(f\"   Total words: {data.get('metadata', {}).get('total_words', 0):,}\")
"
echo ""

# Step 3: Generate embeddings
echo "🔮 Step 3: Generating embeddings & building FAISS index..."
echo "=========================================="
echo "This will chunk documents and generate embeddings using OpenAI"
echo ""

if [ ! -d "../data/vector_store" ]; then
    python ingest_docs.py \
        --input ../data/zuora_docs.json \
        --output ../data/vector_store \
        --provider openai \
        --chunk-size 800 \
        --chunk-overlap 100 \
        --batch-size 100
else
    echo "ℹ️  Vector store already exists at ../data/vector_store"
    echo "   Delete it to re-ingest, or press Enter to continue..."
    read
fi

echo ""
echo "✅ Ingestion complete"
echo ""

# Show ingestion summary
if [ -f "../data/vector_store/ingestion_summary.txt" ]; then
    echo "📊 Ingestion summary:"
    head -15 ../data/vector_store/ingestion_summary.txt | sed 's/^/   /'
    echo ""
fi

# Step 4: Test queries
echo "🔍 Step 4: Testing vector store with sample queries..."
echo "=========================================="
echo ""

queries=(
    "What is Zuora Billing?"
    "How do I configure payment methods?"
    "What is subscription management?"
)

for query in "${queries[@]}"; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Query: \"$query\""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    python query_index.py \
        --index ../data/vector_store \
        --query "$query" \
        --top-k 3 \
        --provider openai

    echo ""
    echo "Press Enter for next query..."
    read
done

echo ""
echo "=========================================="
echo "✅ Demo Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Review the results above"
echo "  2. Try your own queries:"
echo "     python query_index.py --query 'Your question here'"
echo "  3. Integrate with FastAPI backend:"
echo "     See backend/rag.py for implementation"
echo ""
echo "📚 Documentation:"
echo "   - Scraping: scripts/README_SCRAPING.md"
echo "   - Ingestion: scripts/README_INGESTION.md"
echo "   - Main: README.md"
echo ""

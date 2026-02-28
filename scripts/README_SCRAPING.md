# Zuora Documentation Scraping Guide

## Overview

The `scrape_zuora_docs.py` script scrapes publicly accessible Zuora documentation and saves it in JSON format for RAG embedding.

## Features

- ✅ Respectful crawling with delays
- ✅ Automatic content extraction and cleaning
- ✅ HTML to clean text conversion
- ✅ Metadata extraction (title, URL, timestamp, word count)
- ✅ Progress tracking with tqdm
- ✅ JSON export with summary
- ✅ Duplicate URL prevention
- ✅ Error handling and logging

## Installation

```bash
# Install scraping dependencies
pip install -r requirements-scraper.txt

# Or individual packages
pip install requests beautifulsoup4 lxml tqdm
```

## Usage

### Basic Usage

Scrape up to 100 pages from Zuora Knowledge Center:

```bash
python scrape_zuora_docs.py
```

### Custom Options

```bash
# Scrape more pages
python scrape_zuora_docs.py --max-pages 500

# Custom output location
python scrape_zuora_docs.py --output ../data/zuora_knowledge.json

# Increase delay (be more respectful)
python scrape_zuora_docs.py --delay 2.0

# Start from specific URLs
python scrape_zuora_docs.py --start-urls \
    https://knowledgecenter.zuora.com/Billing \
    https://knowledgecenter.zuora.com/Central_Platform
```

### All Options

```bash
python scrape_zuora_docs.py \
    --url https://knowledgecenter.zuora.com \
    --max-pages 200 \
    --delay 1.5 \
    --output ../data/zuora_docs.json \
    --start-urls https://knowledgecenter.zuora.com/API
```

## Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--url` | `https://knowledgecenter.zuora.com` | Base URL to scrape |
| `--max-pages` | `100` | Maximum pages to scrape |
| `--delay` | `1.0` | Delay between requests (seconds) |
| `--output` | `../data/zuora_docs.json` | Output JSON file |
| `--start-urls` | Base URL | Specific starting URLs |

## Output Format

### JSON Structure

```json
{
  "metadata": {
    "source": "Zuora Documentation",
    "base_url": "https://knowledgecenter.zuora.com",
    "total_documents": 150,
    "scraped_at": "2024-02-16T10:30:00",
    "total_words": 45000
  },
  "documents": [
    {
      "content": "Cleaned text content with markdown structure...",
      "metadata": {
        "url": "https://knowledgecenter.zuora.com/page",
        "title": "Page Title",
        "scraped_at": "2024-02-16T10:30:00",
        "word_count": 1500,
        "description": "Page description"
      }
    }
  ]
}
```

### Summary File

A `.summary.txt` file is also generated with:
- Total documents and words
- List of all scraped pages with word counts
- URLs for each document

## What Gets Scraped

### Included
- Main documentation content
- Article pages
- Guide pages
- API documentation
- Knowledge base articles

### Excluded
- Login/logout pages
- Search pages
- Binary files (PDF, images)
- API endpoints (raw data)
- Navigation elements
- Footers and headers

## Content Extraction

The scraper:

1. **Fetches** HTML pages with proper headers
2. **Cleans** by removing scripts, styles, navigation
3. **Extracts** main content area (article, main, .content)
4. **Structures** headings, paragraphs, code blocks
5. **Filters** short or empty content
6. **Saves** with metadata

## Best Practices

### Be Respectful

```bash
# Use appropriate delays (1-2 seconds)
python scrape_zuora_docs.py --delay 2.0

# Don't scrape excessively
python scrape_zuora_docs.py --max-pages 100
```

### Check robots.txt

Before scraping, check:
```
https://knowledgecenter.zuora.com/robots.txt
```

### Monitor Progress

The scraper shows:
- Progress bar with page count
- Documents extracted
- Queue size
- Logs for each page

### Handle Errors

The script handles:
- Network timeouts
- Invalid URLs
- Missing content
- Rate limiting

## Example Workflow

### 1. Scrape Documentation

```bash
cd scripts
python scrape_zuora_docs.py --max-pages 200 --delay 1.5
```

### 2. Check Output

```bash
# View summary
cat ../data/zuora_docs.summary.txt

# Check JSON structure
head -50 ../data/zuora_docs.json
```

### 3. Process for Embedding

```python
import json

# Load scraped docs
with open("../data/zuora_docs.json") as f:
    data = json.load(f)

# Process each document
for doc in data["documents"]:
    content = doc["content"]
    metadata = doc["metadata"]

    # Chunk and embed
    # ... your embedding logic
```

## Troubleshooting

### Connection Errors

```bash
# Increase timeout or add retry logic
# The script has built-in error handling
```

### Empty Content

```bash
# Some pages may have no extractable content
# Check the logs for skipped pages
```

### Rate Limiting

```bash
# Increase delay if getting blocked
python scrape_zuora_docs.py --delay 3.0
```

### Memory Issues

```bash
# Reduce max pages
python scrape_zuora_docs.py --max-pages 50
```

## Advanced Usage

### Target Specific Sections

```bash
# Only scrape billing docs
python scrape_zuora_docs.py \
    --start-urls https://knowledgecenter.zuora.com/Billing \
    --max-pages 100
```

### Multiple Runs

```bash
# Scrape different sections separately
python scrape_zuora_docs.py --start-urls \
    https://knowledgecenter.zuora.com/Billing \
    --output ../data/zuora_billing.json

python scrape_zuora_docs.py --start-urls \
    https://knowledgecenter.zuora.com/API \
    --output ../data/zuora_api.json
```

### Merge Results

```python
import json

# Load multiple JSON files
files = ["zuora_billing.json", "zuora_api.json"]
all_docs = []

for file in files:
    with open(file) as f:
        data = json.load(f)
        all_docs.extend(data["documents"])

# Save merged
with open("zuora_all.json", "w") as f:
    json.dump({"documents": all_docs}, f)
```

## Next Steps

After scraping:

1. **Process Documents**: Chunk text for embedding
2. **Generate Embeddings**: Use OpenAI or similar
3. **Store Vectors**: Add to ChromaDB/Pinecone
4. **Test Retrieval**: Query the vector store

See `../backend/embeddings.py` for processing pipeline.

## Ethical Considerations

- ✅ Respect robots.txt
- ✅ Use reasonable delays
- ✅ Identify your bot in headers
- ✅ Only scrape public content
- ✅ Don't overload servers
- ✅ Cache results to avoid re-scraping

## License

For educational and research purposes. Ensure you have permission to scrape and use the content according to Zuora's terms of service.

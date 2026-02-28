#!/usr/bin/env python3
"""
Test script for Zuora documentation scraper.
Runs a quick test with limited pages to verify functionality.
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from scrape_zuora_docs import ZuoraDocScraper


def test_scraper():
    """Run a quick test of the scraper."""
    print("Testing Zuora Documentation Scraper...")
    print("=" * 50)

    # Create scraper with minimal settings
    scraper = ZuoraDocScraper(
        base_url="https://knowledgecenter.zuora.com",
        max_pages=3,  # Only scrape 3 pages for testing
        delay=1.0,
        output_file="../data/test_docs.json",
    )

    # Run scraping
    print("\n1. Starting test scrape (3 pages max)...")
    documents = scraper.scrape()

    # Check results
    print(f"\n2. Scraped {len(documents)} documents")

    if documents:
        print("\n3. Sample document:")
        doc = documents[0]
        print(f"   Title: {doc['metadata']['title']}")
        print(f"   URL: {doc['metadata']['url']}")
        print(f"   Words: {doc['metadata']['word_count']}")
        print(f"   Content preview: {doc['content'][:200]}...")

        # Save documents
        print("\n4. Saving documents...")
        scraper.save_documents()
        print(f"   Saved to: {scraper.output_file}")

        print("\n✅ Test completed successfully!")
        return True
    else:
        print("\n❌ No documents were scraped. Check the URL and network connection.")
        return False


if __name__ == "__main__":
    success = test_scraper()
    sys.exit(0 if success else 1)

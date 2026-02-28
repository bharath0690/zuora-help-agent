#!/usr/bin/env python3
"""
Zuora Documentation Scraper
Scrapes publicly accessible Zuora documentation and saves as JSON for embedding.

Usage:
    python scrape_zuora_docs.py --output ../data/zuora_docs.json
    python scrape_zuora_docs.py --url https://knowledgecenter.zuora.com --max-pages 100
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional, Set
from urllib.parse import urljoin, urlparse
from datetime import datetime

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ZuoraDocScraper:
    """
    Scraper for Zuora documentation pages.
    Handles crawling, text extraction, and JSON export.
    """

    def __init__(
        self,
        base_url: str = "https://docs.zuora.com/en",
        max_pages: int = 100,
        delay: float = 1.0,
        output_file: str = "../data/zuora_docs.json",
    ):
        """
        Initialize the Zuora documentation scraper.

        Args:
            base_url: Base URL of Zuora documentation
            max_pages: Maximum number of pages to scrape
            delay: Delay between requests (seconds) - be respectful!
            output_file: Output JSON file path
        """
        self.base_url = base_url
        self.max_pages = max_pages
        self.delay = delay
        self.output_file = Path(output_file)
        self.visited_urls: Set[str] = set()
        self.documents: List[Dict] = []

        # Request headers
        self.headers = {
            "User-Agent": "Mozilla/5.0 (compatible; ZuoraDocBot/1.0; Educational/Research Purpose)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }

        # Create output directory if it doesn't exist
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

    def is_valid_url(self, url: str) -> bool:
        """
        Check if URL is valid and should be scraped.

        Args:
            url: URL to validate

        Returns:
            True if URL should be scraped
        """
        parsed = urlparse(url)

        # Must be same domain
        base_parsed = urlparse(self.base_url)
        if parsed.netloc != base_parsed.netloc:
            return False

        # Skip certain paths
        skip_patterns = [
            "/search",
            "/login",
            "/logout",
            "/download",
            ".pdf",
            ".zip",
            ".jpg",
            ".png",
            "/api/v1",  # Skip raw API endpoints
        ]

        for pattern in skip_patterns:
            if pattern in url.lower():
                return False

        return True

    def fetch_page(self, url: str) -> Optional[BeautifulSoup]:
        """
        Fetch and parse a single page.

        Args:
            url: URL to fetch

        Returns:
            BeautifulSoup object or None if failed
        """
        try:
            logger.info(f"Fetching: {url}")
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()

            # Check if it's HTML
            content_type = response.headers.get("Content-Type", "")
            if "text/html" not in content_type:
                logger.warning(f"Skipping non-HTML content: {url}")
                return None

            return BeautifulSoup(response.content, "html.parser")

        except requests.RequestException as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            return None

    def extract_text(self, soup: BeautifulSoup, url: str) -> Optional[Dict]:
        """
        Extract meaningful text from parsed HTML.

        Args:
            soup: BeautifulSoup object
            url: Source URL

        Returns:
            Dictionary with extracted content
        """
        try:
            # Remove unwanted elements
            for element in soup(["script", "style", "nav", "footer", "iframe", "noscript"]):
                element.decompose()

            # Try to find main content area (common patterns)
            main_content = None
            for selector in [
                "main",
                "article",
                ".content",
                ".main-content",
                "#content",
                ".documentation",
                ".doc-content",
            ]:
                main_content = soup.select_one(selector)
                if main_content:
                    break

            # Fall back to body if no main content found
            if not main_content:
                main_content = soup.body

            if not main_content:
                logger.warning(f"No content found for {url}")
                return None

            # Extract title
            title = None
            if soup.title:
                title = soup.title.string.strip()
            else:
                h1 = soup.find("h1")
                title = h1.get_text(strip=True) if h1 else "Untitled"

            # Extract text with some structure
            text_parts = []

            # Get all text elements with their tags
            for element in main_content.find_all(["h1", "h2", "h3", "h4", "p", "li", "td", "code", "pre"]):
                text = element.get_text(strip=True)
                if text and len(text) > 10:  # Filter out very short text
                    # Add tag context for better chunking later
                    if element.name in ["h1", "h2", "h3", "h4"]:
                        text_parts.append(f"\n## {text}\n")
                    elif element.name == "code" or element.name == "pre":
                        text_parts.append(f"\n```\n{text}\n```\n")
                    else:
                        text_parts.append(text)

            content = "\n".join(text_parts)

            # Clean up whitespace
            content = "\n".join(line.strip() for line in content.split("\n") if line.strip())

            # Extract metadata
            metadata = {
                "url": url,
                "title": title,
                "scraped_at": datetime.now().isoformat(),
                "word_count": len(content.split()),
            }

            # Try to extract description/summary
            meta_desc = soup.find("meta", attrs={"name": "description"})
            if meta_desc and meta_desc.get("content"):
                metadata["description"] = meta_desc["content"].strip()

            return {
                "content": content,
                "metadata": metadata,
            }

        except Exception as e:
            logger.error(f"Error extracting text from {url}: {str(e)}")
            return None

    def find_links(self, soup: BeautifulSoup, current_url: str) -> List[str]:
        """
        Find all valid links on the page.

        Args:
            soup: BeautifulSoup object
            current_url: Current page URL for resolving relative links

        Returns:
            List of valid URLs to scrape
        """
        links = []

        for link in soup.find_all("a", href=True):
            href = link["href"]

            # Resolve relative URLs
            absolute_url = urljoin(current_url, href)

            # Remove fragment
            absolute_url = absolute_url.split("#")[0]

            # Validate and add
            if self.is_valid_url(absolute_url) and absolute_url not in self.visited_urls:
                links.append(absolute_url)

        return links

    def scrape(self, start_urls: Optional[List[str]] = None) -> List[Dict]:
        """
        Main scraping method - crawls documentation and extracts content.

        Args:
            start_urls: Initial URLs to start crawling (default: base_url)

        Returns:
            List of extracted documents
        """
        if start_urls is None:
            start_urls = [self.base_url]

        to_visit = start_urls.copy()
        progress_bar = tqdm(total=self.max_pages, desc="Scraping pages")

        while to_visit and len(self.visited_urls) < self.max_pages:
            url = to_visit.pop(0)

            # Skip if already visited
            if url in self.visited_urls:
                continue

            # Mark as visited
            self.visited_urls.add(url)

            # Fetch page
            soup = self.fetch_page(url)
            if not soup:
                continue

            # Extract content
            doc = self.extract_text(soup, url)
            if doc and len(doc["content"]) > 100:  # Only save pages with substantial content
                self.documents.append(doc)
                logger.info(
                    f"Extracted {doc['metadata']['word_count']} words from: {doc['metadata']['title']}"
                )

            # Find new links to visit
            new_links = self.find_links(soup, url)
            to_visit.extend(new_links)

            # Update progress
            progress_bar.update(1)
            progress_bar.set_postfix(
                {"docs": len(self.documents), "queue": len(to_visit)}
            )

            # Be respectful - add delay
            time.sleep(self.delay)

        progress_bar.close()
        logger.info(f"Scraping complete. Extracted {len(self.documents)} documents.")

        return self.documents

    def save_documents(self) -> None:
        """Save extracted documents to JSON file."""
        try:
            output_data = {
                "metadata": {
                    "source": "Zuora Documentation",
                    "base_url": self.base_url,
                    "total_documents": len(self.documents),
                    "scraped_at": datetime.now().isoformat(),
                    "total_words": sum(doc["metadata"]["word_count"] for doc in self.documents),
                },
                "documents": self.documents,
            }

            with open(self.output_file, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved {len(self.documents)} documents to {self.output_file}")

            # Also save a summary
            summary_file = self.output_file.with_suffix(".summary.txt")
            with open(summary_file, "w", encoding="utf-8") as f:
                f.write(f"Zuora Documentation Scrape Summary\n")
                f.write(f"{'=' * 50}\n\n")
                f.write(f"Total Documents: {len(self.documents)}\n")
                f.write(f"Total Words: {sum(doc['metadata']['word_count'] for doc in self.documents):,}\n")
                f.write(f"Base URL: {self.base_url}\n")
                f.write(f"Scraped At: {datetime.now().isoformat()}\n\n")
                f.write(f"Documents:\n")
                for i, doc in enumerate(self.documents, 1):
                    f.write(
                        f"{i}. {doc['metadata']['title']} "
                        f"({doc['metadata']['word_count']} words)\n"
                        f"   {doc['metadata']['url']}\n"
                    )

            logger.info(f"Saved summary to {summary_file}")

        except Exception as e:
            logger.error(f"Error saving documents: {str(e)}")


def main():
    """Main entry point for the scraper."""
    parser = argparse.ArgumentParser(description="Scrape Zuora documentation for RAG embedding")

    parser.add_argument(
        "--url",
        type=str,
        default="https://docs.zuora.com/en",
        help="Base URL to scrape (default: docs.zuora.com)",
    )

    parser.add_argument(
        "--max-pages",
        type=int,
        default=100,
        help="Maximum number of pages to scrape (default: 100)",
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between requests in seconds (default: 1.0)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="../data/zuora_docs.json",
        help="Output JSON file path (default: ../data/zuora_docs.json)",
    )

    parser.add_argument(
        "--start-urls",
        type=str,
        nargs="+",
        help="Specific URLs to start scraping from",
    )

    args = parser.parse_args()

    # Initialize scraper
    scraper = ZuoraDocScraper(
        base_url=args.url,
        max_pages=args.max_pages,
        delay=args.delay,
        output_file=args.output,
    )

    # Run scraping
    logger.info(f"Starting scrape of {args.url}")
    logger.info(f"Max pages: {args.max_pages}")
    logger.info(f"Delay: {args.delay}s")

    scraper.scrape(start_urls=args.start_urls)
    scraper.save_documents()

    logger.info("Scraping complete!")


if __name__ == "__main__":
    main()

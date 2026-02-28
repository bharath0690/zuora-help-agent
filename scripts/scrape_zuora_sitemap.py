#!/usr/bin/env python3
"""
Zuora Documentation Scraper - Sitemap-based Version
Uses sitemap.xml for efficient crawling of docs.zuora.com

Usage:
    python scrape_zuora_sitemap.py --product billing --max-pages 100
    python scrape_zuora_sitemap.py --all-products --max-pages 500
"""

import argparse
import json
import logging
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Optional
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


class ZuoraSitemapScraper:
    """
    Scraper for Zuora documentation using sitemap.xml for URL discovery.
    More efficient than crawling - gets all URLs upfront.
    """

    # Available Zuora products in docs
    PRODUCTS = {
        "billing": "https://docs.zuora.com/en/zuora-billing/sitemap.xml",
        "payments": "https://docs.zuora.com/en/zuora-payments/sitemap.xml",
        "platform": "https://docs.zuora.com/en/zuora-platform/sitemap.xml",
        "cpq": "https://docs.zuora.com/en/zuora-cpq/sitemap.xml",
        "revenue": "https://docs.zuora.com/en/zuora-revenue/sitemap.xml",
        "ar": "https://docs.zuora.com/en/accounts-receivable/sitemap.xml",
        "zephr": "https://docs.zuora.com/en/zephr/sitemap.xml",
        "basics": "https://docs.zuora.com/en/basics/sitemap.xml",
        "entitlements": "https://docs.zuora.com/en/entitlements/sitemap.xml",
        "release-notes": "https://docs.zuora.com/en/release-notes/sitemap.xml",
    }

    def __init__(
        self,
        products: List[str] = None,
        max_pages: int = 100,
        delay: float = 0.5,
        output_file: str = "../data/zuora_docs.json",
    ):
        """
        Initialize the sitemap-based scraper.

        Args:
            products: List of product names to scrape (or None for all)
            max_pages: Maximum number of pages to scrape
            delay: Delay between requests (seconds)
            output_file: Output JSON file path
        """
        self.products = products or list(self.PRODUCTS.keys())
        self.max_pages = max_pages
        self.delay = delay
        self.output_file = Path(output_file)
        self.documents: List[Dict] = []

        # Request headers
        self.headers = {
            "User-Agent": "Mozilla/5.0 (compatible; ZuoraDocBot/2.0; Educational Purpose)",
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9",
        }

        # Create output directory
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

    def fetch_sitemap_urls(self, sitemap_url: str) -> List[str]:
        """
        Fetch URLs from a sitemap.

        Args:
            sitemap_url: URL of the sitemap XML

        Returns:
            List of page URLs
        """
        try:
            logger.info(f"Fetching sitemap: {sitemap_url}")
            response = requests.get(sitemap_url, headers=self.headers, timeout=10)
            response.raise_for_status()

            # Parse XML
            root = ET.fromstring(response.content)

            # Extract URLs (handle namespace)
            namespace = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}
            urls = [loc.text for loc in root.findall(".//ns:loc", namespace)]

            logger.info(f"Found {len(urls)} URLs in sitemap")
            return urls

        except Exception as e:
            logger.error(f"Error fetching sitemap {sitemap_url}: {str(e)}")
            return []

    def get_all_urls(self) -> List[str]:
        """
        Get all URLs from selected product sitemaps.

        Returns:
            List of all documentation URLs to scrape
        """
        all_urls = []

        for product in self.products:
            if product not in self.PRODUCTS:
                logger.warning(f"Unknown product: {product}. Skipping.")
                continue

            sitemap_url = self.PRODUCTS[product]
            urls = self.fetch_sitemap_urls(sitemap_url)
            all_urls.extend(urls)

            logger.info(f"{product}: {len(urls)} pages")

        logger.info(f"Total URLs to scrape: {len(all_urls)}")
        return all_urls

    def extract_content(self, soup: BeautifulSoup, url: str) -> Optional[Dict]:
        """
        Extract content from a Zuora docs page.

        Args:
            soup: BeautifulSoup object
            url: Source URL

        Returns:
            Dictionary with extracted content and metadata
        """
        try:
            # Remove unwanted elements
            for element in soup(["script", "style", "nav", "footer", "iframe", "noscript", "header"]):
                element.decompose()

            # Find main content - docs.zuora.com specific selectors
            main_content = None
            for selector in [
                "article",
                "main",
                ".content",
                ".doc-content",
                ".article-content",
                "[role='main']",
            ]:
                main_content = soup.select_one(selector)
                if main_content:
                    break

            if not main_content:
                main_content = soup.body

            if not main_content:
                return None

            # Extract title
            title = None
            if soup.title:
                title = soup.title.string.strip()
                # Remove common suffixes
                title = title.replace(" | Zuora", "").replace(" - Zuora Product Documentation", "").strip()
            else:
                h1 = soup.find("h1")
                title = h1.get_text(strip=True) if h1 else "Untitled"

            # Extract structured content
            text_parts = []

            for element in main_content.find_all(
                ["h1", "h2", "h3", "h4", "h5", "p", "li", "td", "code", "pre", "div"]
            ):
                # Skip if parent is code/pre (avoid duplication)
                if element.parent.name in ["code", "pre"]:
                    continue

                text = element.get_text(strip=True)

                if not text or len(text) < 10:
                    continue

                # Add structural markers
                if element.name in ["h1", "h2", "h3"]:
                    text_parts.append(f"\n{'#' * int(element.name[1])} {text}\n")
                elif element.name in ["code", "pre"]:
                    text_parts.append(f"\n```\n{text}\n```\n")
                else:
                    text_parts.append(text)

            content = "\n".join(text_parts)

            # Clean whitespace
            content = "\n".join(line.strip() for line in content.split("\n") if line.strip())

            # Skip if too short
            if len(content) < 100:
                logger.warning(f"Skipping short content: {url}")
                return None

            # Extract product from URL
            product = "unknown"
            for prod_name in self.PRODUCTS.keys():
                if prod_name in url or prod_name.replace("-", " ") in url:
                    product = prod_name
                    break

            # Build metadata
            metadata = {
                "url": url,
                "title": title,
                "product": product,
                "scraped_at": datetime.now().isoformat(),
                "word_count": len(content.split()),
                "source": "docs.zuora.com",
            }

            # Try to extract description
            meta_desc = soup.find("meta", attrs={"name": "description"})
            if meta_desc and meta_desc.get("content"):
                metadata["description"] = meta_desc["content"].strip()

            return {"content": content, "metadata": metadata}

        except Exception as e:
            logger.error(f"Error extracting content from {url}: {str(e)}")
            return None

    def fetch_page(self, url: str) -> Optional[BeautifulSoup]:
        """
        Fetch and parse a page.

        Args:
            url: URL to fetch

        Returns:
            BeautifulSoup object or None
        """
        try:
            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
            return BeautifulSoup(response.content, "html.parser")

        except Exception as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            return None

    def scrape(self) -> List[Dict]:
        """
        Main scraping method using sitemap URLs.

        Returns:
            List of extracted documents
        """
        # Get all URLs from sitemaps
        urls = self.get_all_urls()

        # Limit to max_pages
        urls = urls[: self.max_pages]

        logger.info(f"Scraping {len(urls)} pages...")

        # Progress bar
        progress_bar = tqdm(urls, desc="Scraping docs")

        for url in progress_bar:
            # Fetch page
            soup = self.fetch_page(url)
            if not soup:
                continue

            # Extract content
            doc = self.extract_content(soup, url)
            if doc:
                self.documents.append(doc)
                progress_bar.set_postfix(
                    {
                        "docs": len(self.documents),
                        "words": doc["metadata"]["word_count"],
                    }
                )

            # Delay
            time.sleep(self.delay)

        logger.info(f"Scraping complete. Extracted {len(self.documents)} documents.")
        return self.documents

    def save_documents(self) -> None:
        """Save extracted documents to JSON."""
        try:
            # Calculate statistics
            total_words = sum(doc["metadata"]["word_count"] for doc in self.documents)
            products_count = {}
            for doc in self.documents:
                prod = doc["metadata"]["product"]
                products_count[prod] = products_count.get(prod, 0) + 1

            output_data = {
                "metadata": {
                    "source": "Zuora Documentation (docs.zuora.com)",
                    "products": self.products,
                    "total_documents": len(self.documents),
                    "total_words": total_words,
                    "products_breakdown": products_count,
                    "scraped_at": datetime.now().isoformat(),
                },
                "documents": self.documents,
            }

            # Save JSON
            with open(self.output_file, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved {len(self.documents)} documents to {self.output_file}")

            # Save summary
            summary_file = self.output_file.with_suffix(".summary.txt")
            with open(summary_file, "w", encoding="utf-8") as f:
                f.write("Zuora Documentation Scrape Summary\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Source: docs.zuora.com\n")
                f.write(f"Products: {', '.join(self.products)}\n")
                f.write(f"Total Documents: {len(self.documents)}\n")
                f.write(f"Total Words: {total_words:,}\n")
                f.write(f"Scraped At: {datetime.now().isoformat()}\n\n")

                f.write("Products Breakdown:\n")
                for prod, count in sorted(products_count.items()):
                    f.write(f"  - {prod}: {count} pages\n")

                f.write("\nDocuments:\n")
                for i, doc in enumerate(self.documents, 1):
                    f.write(
                        f"{i}. [{doc['metadata']['product']}] {doc['metadata']['title']} "
                        f"({doc['metadata']['word_count']} words)\n"
                        f"   {doc['metadata']['url']}\n"
                    )

            logger.info(f"Saved summary to {summary_file}")

        except Exception as e:
            logger.error(f"Error saving documents: {str(e)}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Scrape Zuora documentation using sitemaps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available products:
  {', '.join(ZuoraSitemapScraper.PRODUCTS.keys())}

Examples:
  # Scrape billing docs
  python scrape_zuora_sitemap.py --product billing --max-pages 100

  # Scrape multiple products
  python scrape_zuora_sitemap.py --product billing cpq platform --max-pages 300

  # Scrape all products
  python scrape_zuora_sitemap.py --all-products --max-pages 1000
        """,
    )

    parser.add_argument(
        "--product",
        nargs="+",
        choices=list(ZuoraSitemapScraper.PRODUCTS.keys()),
        help="Products to scrape (space-separated)",
    )

    parser.add_argument(
        "--all-products",
        action="store_true",
        help="Scrape all available products",
    )

    parser.add_argument(
        "--max-pages",
        type=int,
        default=100,
        help="Maximum pages to scrape (default: 100)",
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between requests in seconds (default: 0.5)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="../data/zuora_docs.json",
        help="Output JSON file (default: ../data/zuora_docs.json)",
    )

    args = parser.parse_args()

    # Determine products
    if args.all_products:
        products = None  # Will use all products
    elif args.product:
        products = args.product
    else:
        parser.error("Specify --product or --all-products")

    # Initialize scraper
    scraper = ZuoraSitemapScraper(
        products=products,
        max_pages=args.max_pages,
        delay=args.delay,
        output_file=args.output,
    )

    # Run
    logger.info("Starting Zuora documentation scrape...")
    scraper.scrape()
    scraper.save_documents()
    logger.info("Complete!")


if __name__ == "__main__":
    main()

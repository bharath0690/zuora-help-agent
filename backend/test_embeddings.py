#!/usr/bin/env python3
"""
Test script for embeddings.py

Demonstrates:
- Embedding generation with different providers
- Text chunking
- Error handling and retries
- Batch processing

Usage:
    # Set API key
    export VOYAGE_API_KEY="pa-..."  # or OPENAI_API_KEY

    # Run tests
    python test_embeddings.py
"""

import asyncio
import os
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from embeddings import EmbeddingGenerator, DocumentChunker, EmbeddingError


async def test_embedding_generation():
    """Test basic embedding generation."""
    print("\n" + "=" * 60)
    print("Test 1: Embedding Generation")
    print("=" * 60)

    try:
        # Initialize with Voyage AI (or OpenAI if Voyage key not available)
        provider = "voyage" if os.getenv("VOYAGE_API_KEY") else "openai"
        print(f"\nUsing provider: {provider}")

        embedder = EmbeddingGenerator(provider=provider)

        # Test texts
        texts = [
            "Zuora CPQ is a configure, price, quote solution for subscription businesses.",
            "The Zuora platform enables companies to launch and manage subscription services.",
            "Payment processing in Zuora supports multiple payment gateways and methods.",
        ]

        print(f"\nGenerating embeddings for {len(texts)} texts...")

        # Generate embeddings
        embeddings = await embedder.generate_embeddings(texts)

        print(f"\n✅ Success!")
        print(f"   Shape: {embeddings.shape}")
        print(f"   Dimension: {embedder.get_dimension()}")
        print(f"   Provider info: {embedder.get_provider_info()}")

        # Show sample embedding
        print(f"\n   Sample embedding (first 10 values):")
        print(f"   {embeddings[0][:10]}")

        return True

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return False


async def test_query_embedding():
    """Test query-specific embedding."""
    print("\n" + "=" * 60)
    print("Test 2: Query Embedding")
    print("=" * 60)

    try:
        provider = "voyage" if os.getenv("VOYAGE_API_KEY") else "openai"
        embedder = EmbeddingGenerator(provider=provider)

        query = "How do I configure SSO with Zuora?"
        print(f"\nQuery: \"{query}\"")

        # Generate query embedding
        embedding = await embedder.generate_query_embedding(query)

        print(f"\n✅ Success!")
        print(f"   Shape: {embedding.shape}")
        print(f"   First 10 values: {embedding[:10]}")

        return True

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return False


async def test_batching():
    """Test batch processing."""
    print("\n" + "=" * 60)
    print("Test 3: Batch Processing")
    print("=" * 60)

    try:
        provider = "voyage" if os.getenv("VOYAGE_API_KEY") else "openai"
        embedder = EmbeddingGenerator(provider=provider, batch_size=2)

        # Create more texts to test batching
        texts = [
            f"This is test text number {i} about Zuora subscriptions."
            for i in range(5)
        ]

        print(f"\nGenerating embeddings for {len(texts)} texts with batch_size=2...")

        embeddings = await embedder.generate_embeddings(texts)

        print(f"\n✅ Success!")
        print(f"   Total embeddings: {len(embeddings)}")
        print(f"   Processed in {(len(texts) + 1) // 2} batches")

        return True

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return False


def test_chunking():
    """Test document chunking."""
    print("\n" + "=" * 60)
    print("Test 4: Document Chunking")
    print("=" * 60)

    try:
        # Initialize chunker
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)

        # Sample document
        document = """
        Zuora CPQ Overview

        Zuora CPQ (Configure, Price, Quote) is a comprehensive solution for managing
        complex subscription pricing and quoting processes. It enables sales teams to
        quickly create accurate quotes for subscription-based products and services.

        Key Features

        Product Catalog: Define subscription products, features, and pricing models.
        Configure products with flexible options and dependencies.

        Pricing Rules: Set up dynamic pricing based on quantity, term, or custom rules.
        Support tiered pricing, volume discounts, and promotional pricing.

        Quote Generation: Generate professional quotes with automated calculations.
        Support for multi-product bundles and complex subscription terms.

        Integration: Seamlessly integrate with Zuora Billing for order management.
        Connect with CRM systems like Salesforce for streamlined workflows.
        """

        print(f"\nChunking document ({len(document)} chars)...")

        metadata = {
            "source": "zuora_cpq_guide.md",
            "product": "cpq",
        }

        chunks = chunker.chunk_text(document, metadata)

        print(f"\n✅ Success!")
        print(f"   Total chunks: {len(chunks)}")

        for i, chunk in enumerate(chunks):
            print(f"\n   Chunk {i + 1}:")
            print(f"      Tokens: {chunk['metadata']['token_count']}")
            print(f"      Preview: {chunk['content'][:80]}...")

        return True

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return False


async def test_error_handling():
    """Test error handling with invalid input."""
    print("\n" + "=" * 60)
    print("Test 5: Error Handling")
    print("=" * 60)

    try:
        provider = "voyage" if os.getenv("VOYAGE_API_KEY") else "openai"
        embedder = EmbeddingGenerator(provider=provider)

        print("\nTest 5a: Empty text list...")
        embeddings = await embedder.generate_embeddings([])
        print(f"   Result: {len(embeddings)} embeddings (expected 0)")

        print("\nTest 5b: Empty query...")
        try:
            await embedder.generate_query_embedding("")
            print("   ❌ Should have raised ValueError")
            return False
        except ValueError as e:
            print(f"   ✅ Correctly raised ValueError: {str(e)}")

        print("\nTest 5c: Invalid provider...")
        try:
            bad_embedder = EmbeddingGenerator(provider="invalid")
            print("   ❌ Should have raised ValueError")
            return False
        except ValueError as e:
            print(f"   ✅ Correctly raised ValueError: {str(e)}")

        return True

    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        return False


async def test_local_embeddings():
    """Test local embeddings (no API key required)."""
    print("\n" + "=" * 60)
    print("Test 6: Local Embeddings (Optional)")
    print("=" * 60)

    try:
        print("\nInitializing local model (may download on first run)...")
        embedder = EmbeddingGenerator(provider="local", model="all-MiniLM-L6-v2")

        texts = ["Zuora billing platform", "Subscription management"]

        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = await embedder.generate_embeddings(texts)

        print(f"\n✅ Success!")
        print(f"   Shape: {embeddings.shape}")
        print(f"   Provider: local (no API calls)")

        return True

    except ImportError:
        print("\n⚠️  Skipped: sentence-transformers not installed")
        print("   Install: pip install sentence-transformers")
        return True  # Not a failure, just skipped

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return False


async def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Embeddings.py Test Suite")
    print("=" * 60)

    # Check for API keys
    has_voyage = bool(os.getenv("VOYAGE_API_KEY"))
    has_openai = bool(os.getenv("OPENAI_API_KEY"))

    print(f"\nAPI Keys:")
    print(f"   Voyage AI: {'✅' if has_voyage else '❌'}")
    print(f"   OpenAI:    {'✅' if has_openai else '❌'}")

    if not has_voyage and not has_openai:
        print("\n⚠️  Warning: No API keys found!")
        print("   Set VOYAGE_API_KEY or OPENAI_API_KEY to run tests")
        print("\n   Example:")
        print("      export VOYAGE_API_KEY='pa-...'")
        print("      python test_embeddings.py")
        return

    # Run tests
    results = []

    results.append(("Embedding Generation", await test_embedding_generation()))
    results.append(("Query Embedding", await test_query_embedding()))
    results.append(("Batch Processing", await test_batching()))
    results.append(("Document Chunking", test_chunking()))
    results.append(("Error Handling", await test_error_handling()))
    results.append(("Local Embeddings", await test_local_embeddings()))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")


if __name__ == "__main__":
    asyncio.run(main())

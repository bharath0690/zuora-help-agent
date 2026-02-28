#!/usr/bin/env python3
"""
Test script for build_index.py

Creates sample processed_docs.json and tests index building.

Usage:
    python test_build_index.py
"""

import json
import asyncio
from pathlib import Path

# Sample processed documents
SAMPLE_DOCS = {
    "metadata": {
        "source": "test_data",
        "created_at": "2024-02-16T10:00:00",
        "total_documents": 5,
    },
    "documents": [
        {
            "content": "Zuora CPQ (Configure, Price, Quote) is a comprehensive solution for managing subscription pricing and quoting processes. It enables sales teams to create accurate quotes.",
            "metadata": {
                "source": "zuora_cpq_guide.md",
                "title": "Zuora CPQ Overview",
                "product": "cpq",
                "chunk_index": 0,
            }
        },
        {
            "content": "Zuora Billing is the core platform for subscription billing and revenue management. It automates the entire quote-to-cash process for subscription businesses.",
            "metadata": {
                "source": "zuora_billing_intro.md",
                "title": "Zuora Billing Introduction",
                "product": "billing",
                "chunk_index": 0,
            }
        },
        {
            "content": "Payment methods in Zuora support credit cards, ACH, PayPal, and custom payment gateways. Configure payment settings in the Zuora admin console.",
            "metadata": {
                "source": "payment_methods.md",
                "title": "Payment Methods Configuration",
                "product": "payments",
                "chunk_index": 0,
            }
        },
        {
            "content": "SSO (Single Sign-On) can be configured using SAML 2.0 or OpenID Connect. Navigate to Settings > Security > Single Sign-On to configure your identity provider.",
            "metadata": {
                "source": "sso_setup.md",
                "title": "SSO Configuration Guide",
                "product": "platform",
                "chunk_index": 0,
            }
        },
        {
            "content": "Subscription amendments allow you to modify existing subscriptions. Common amendments include upgrades, downgrades, and adding or removing products.",
            "metadata": {
                "source": "subscription_amendments.md",
                "title": "Subscription Amendments",
                "product": "billing",
                "chunk_index": 0,
            }
        },
    ]
}


async def main():
    """Test index building."""
    print("=" * 60)
    print("Build Index Test")
    print("=" * 60)

    # 1. Create sample processed_docs.json
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    test_input = data_dir / "processed_docs.json"
    print(f"\n1. Creating sample data: {test_input}")

    with open(test_input, 'w') as f:
        json.dump(SAMPLE_DOCS, f, indent=2)

    print(f"   ✅ Created {len(SAMPLE_DOCS['documents'])} sample documents")

    # 2. Run build_index.py
    print(f"\n2. Building FAISS index...")
    print(f"   Run: python build_index.py")
    print(f"\n   Note: You need to run build_index.py separately")
    print(f"   It requires an API key (VOYAGE_API_KEY or OPENAI_API_KEY)")

    # 3. Show expected output
    index_path = data_dir / "faiss.index"
    metadata_path = data_dir / "metadata.json"

    print(f"\n3. Expected output files:")
    print(f"   - {index_path}")
    print(f"   - {metadata_path}")

    # 4. Show runtime usage example
    print(f"\n4. Runtime usage example:")
    print("""
    from faiss_loader import FAISSIndexLoader
    from embeddings import EmbeddingGenerator

    # Load index
    loader = FAISSIndexLoader(
        index_path="../data/faiss.index",
        metadata_path="../data/metadata.json"
    )

    # Generate query embedding
    embedder = EmbeddingGenerator(provider="voyage")
    query = "How do I configure SSO?"
    query_embedding = await embedder.generate_query_embedding(query)

    # Search
    results = loader.search(query_embedding, top_k=3)

    for result in results:
        print(f"Title: {result['title']}")
        print(f"Score: {result['score']:.3f}")
        print(f"Text: {result['chunk_text'][:100]}...")
    """)

    print("\n" + "=" * 60)
    print("✅ Test data created successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Set API key: export VOYAGE_API_KEY='pa-...'")
    print("2. Build index: python build_index.py")
    print("3. Test search: python -c 'from faiss_loader import FAISSIndexLoader; ...'")


if __name__ == "__main__":
    asyncio.run(main())

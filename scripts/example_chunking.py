"""
Exemple de chunking des documents scrapés.

Ce script montre comment :
1. Scraper des jugements
2. Les découper en chunks avec métadonnées
"""
import sys
from pathlib import Path

# Ajouter le répertoire racine du projet au PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import asyncio
from src import (
    EntscheidsucheScraper,
    ScraperConfig,
    chunk_document_by_pages
)


async def main():
    """Pipeline de chunking."""
    print("=== LawLLM - Chunking des documents ===\n")
    
    # Configuration
    config = ScraperConfig(
        output_dir="data/raw",
        rate_limit=5,
        max_concurrent=10
    )
    
    # 1. Scraping
    print("1. Scraping des jugements...")
    documents = []
    async with EntscheidsucheScraper(config) as scraper:
        count = 0
        async for doc in scraper.fetch_spider("VD_FindInfo", max_docs=5):
            documents.append(doc)
            count += 1
            print(f"  ✓ Document {count}: {doc.id} ({len(doc.content)} caractères)")
            await scraper.save_document(doc)
    
    print(f"\n✓ {len(documents)} documents scrapés\n")
    
    if not documents:
        print("Aucun document à traiter. Arrêt.")
        return
    
    # 2. Chunking
    print("2. Découpage en chunks (environ 3 pages par chunk)...")
    all_chunks = []
    
    for doc in documents:
        chunks = chunk_document_by_pages(doc, pages_per_chunk=3)
        all_chunks.extend(chunks)
        print(f"  ✓ {doc.id}: {len(chunks)} chunks créés")
        
        # Afficher un exemple de chunk
        if chunks:
            print(f"    Exemple chunk 0: {chunks[0].text[:100]}...")
    
    print(f"\n✓ {len(all_chunks)} chunks créés au total")
    print(f"\nLes chunks contiennent toutes les métadonnées nécessaires:")
    if all_chunks:
        example = all_chunks[0]
        print(f"  - Document ID: {example.document_id}")
        print(f"  - Spider: {example.spider}")
        print(f"  - Langue: {example.language}")
        print(f"  - Date: {example.date}")
        print(f"  - URL: {example.entscheidsuche_url}")


if __name__ == "__main__":
    asyncio.run(main())


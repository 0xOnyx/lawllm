"""
Exemple d'utilisation du scraper.

Ce script montre comment utiliser le scraper pour télécharger des jugements.
"""
import sys
from pathlib import Path

# Ajouter le répertoire racine du projet au PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import asyncio
from src import EntscheidsucheScraper, ScraperConfig, list_available_spiders


async def main():
    """Exemple principal."""
    print("=== LawLLM - Exemple de scraping ===\n")
    
    # Lister les spiders disponibles
    print("Spiders disponibles:")
    spiders = list_available_spiders()
    for spider_id, description in list(spiders.items())[:5]:  # Afficher les 5 premiers
        print(f"  - {spider_id}: {description}")
    print(f"  ... et {len(spiders) - 5} autres\n")
    
    # Configuration
    config = ScraperConfig(
        output_dir="data/raw",
        rate_limit=5,  # 5 requêtes par seconde
        max_concurrent=10
    )
    
    # Scraper quelques documents du Tribunal fédéral
    print("Scraping de quelques documents du Tribunal fédéral (VD_FindInfo)...")
    async with EntscheidsucheScraper(config) as scraper:
        count = 0
        async for doc in scraper.fetch_spider("VD_FindInfo", max_docs=100):
            count += 1
            print(f"\nDocument {count}:")
            print(f"  ID: {doc.id}")
            print(f"  Titre: {doc.title or 'N/A'}")
            print(f"  Date: {doc.date or 'N/A'}")
            print(f"  Langue: {doc.language}")
            print(f"  Contenu: {len(doc.content)} caractères")
            saved_path = await scraper.save_document(doc)
            print(f"  Sauvegardé dans: {saved_path}")
    
    print(f"\n✓ Scraping terminé: {count} documents récupérés")


if __name__ == "__main__":
    asyncio.run(main())


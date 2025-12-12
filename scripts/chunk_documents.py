"""
Script pour créer des chunks à partir de documents déjà scrapés.

Utile si vous avez déjà des documents dans data/raw et que vous voulez
les découper en chunks sans les re-scraper.
"""
import sys
import json
from pathlib import Path

# Ajouter le répertoire racine du projet au PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src import chunk_document_by_pages, ScrapedDocument


def load_scraped_documents(data_dir: str = "data/raw") -> list[ScrapedDocument]:
    """
    Charge tous les documents scrapés depuis data/raw.
    
    Args:
        data_dir: Répertoire contenant les documents JSON
        
    Returns:
        Liste de ScrapedDocument
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Le répertoire {data_dir} n'existe pas.")
        return []
    
    documents = []
    
    # Parcourir tous les sous-répertoires (spiders)
    for spider_dir in data_path.iterdir():
        if not spider_dir.is_dir():
            continue
        
        spider = spider_dir.name
        print(f"Chargement des documents de {spider}...")
        
        # Charger tous les fichiers JSON
        json_files = list(spider_dir.glob("*.json"))
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    doc = ScrapedDocument(**data)
                    documents.append(doc)
            except Exception as e:
                print(f"  Erreur lors du chargement de {json_file}: {e}")
        
        print(f"  ✓ {len([d for d in documents if d.spider == spider])} documents chargés")
    
    return documents


def save_chunks(chunks, output_dir: str = "data/chunks"):
    """    Sauvegarde les chunks dans des fichiers JSON.
    
    Args:
        chunks: Liste de TextChunk
        output_dir: Répertoire de sortie
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Grouper par document
    chunks_by_doc = {}
    for chunk in chunks:
        doc_id = chunk.document_id
        if doc_id not in chunks_by_doc:
            chunks_by_doc[doc_id] = []
        chunks_by_doc[doc_id].append(chunk)
    
    # Sauvegarder chaque document
    for doc_id, doc_chunks in chunks_by_doc.items():
        filepath = output_path / f"{doc_id}_chunks.json"
        chunks_data = [chunk.model_dump() for chunk in doc_chunks]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
        
        print(f"  ✓ {len(doc_chunks)} chunks sauvegardés pour {doc_id}")


def main():
    """Crée des chunks à partir des documents scrapés."""
    print("=== LawLLM - Création de chunks ===\n")
    
    # Charger les documents
    print("1. Chargement des documents scrapés...")
    documents = load_scraped_documents("data/raw")
    
    if not documents:
        print("Aucun document trouvé dans data/raw")
        print("Utilisez d'abord example_scrape.py pour scraper des documents.")
        return
    
    print(f"\n✓ {len(documents)} documents chargés\n")
    
    # Chunking
    print("2. Découpage en chunks (environ 3 pages par chunk)...")
    all_chunks = []
    
    for doc in documents:
        chunks = chunk_document_by_pages(doc, pages_per_chunk=3)
        all_chunks.extend(chunks)
        if len(all_chunks) % 100 == 0:
            print(f"  {len(all_chunks)} chunks créés...")
        print(f"  ✓ {doc.id}: {len(chunks)} chunks")
    
    print(f"\n✓ {len(all_chunks)} chunks créés au total\n")
    
    # Sauvegarder les chunks
    print("3. Sauvegarde des chunks...")
    save_chunks(all_chunks, "data/chunks")
    print("✓ Chunks sauvegardés dans data/chunks/")


if __name__ == "__main__":
    main()


"""
Exemple d'indexation des résumés dans ChromaDB.

Ce script montre comment :
1. Charger des SummarizedChunk
2. Générer les embeddings
3. Indexer dans ChromaDB
4. Effectuer des recherches
"""
import sys
import json
from pathlib import Path

# Ajouter le répertoire racine du projet au PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src import (
    SummarizedChunk,
    VectorStore,
    load_summarized_chunks,
)


def load_summarized_chunks_from_dir(summaries_dir: str = "data/summaries") -> list[SummarizedChunk]:
    """
    Charge tous les SummarizedChunk depuis un répertoire.
    
    Args:
        summaries_dir: Répertoire contenant les fichiers JSON de résumés
        
    Returns:
        Liste de SummarizedChunk
    """
    summaries_path = Path(summaries_dir)
    if not summaries_path.exists():
        print(f"Le répertoire {summaries_dir} n'existe pas.")
        return []
    
    all_chunks = []
    
    # Charger tous les fichiers JSON
    json_files = list(summaries_path.glob("*_summarized.json"))
    for json_file in json_files:
        try:
            chunks = load_summarized_chunks(str(json_file))
            all_chunks.extend(chunks)
            print(f"  ✓ {len(chunks)} chunks chargés depuis {json_file.name}")
        except Exception as e:
            print(f"  ✗ Erreur lors du chargement de {json_file}: {e}")
    
    return all_chunks


def main():
    """Pipeline d'indexation dans ChromaDB."""
    print("=== LawLLM - Indexation dans ChromaDB ===\n")
    
    # Charger les résumés
    print("1. Chargement des SummarizedChunk...")
    summarized_chunks = load_summarized_chunks_from_dir("data/summaries")
    
    if not summarized_chunks:
        print("Aucun SummarizedChunk trouvé dans data/summaries")
        print("Utilisez d'abord example_summarize.py pour créer des résumés.")
        return
    
    print(f"✓ {len(summarized_chunks)} SummarizedChunk chargés\n")
    
    # Initialiser le VectorStore
    print("2. Initialisation du VectorStore...")
    store = VectorStore(
        collection_name="lawllm_chunks",
        db_path="chroma_db"
    )
    print("✓ VectorStore initialisé\n")
    
    # Afficher les statistiques avant indexation
    print("3. Statistiques avant indexation:")
    stats_before = store.get_collection_stats()
    print(f"  Documents existants: {stats_before['total_documents']}")
    
    # Demander confirmation si des documents existent déjà
    if stats_before['total_documents'] > 0:
        response = input(f"\n⚠ {stats_before['total_documents']} documents existent déjà. Réinitialiser? (o/N): ")
        if response.lower() == 'o':
            store.reset_collection()
            print("✓ Collection réinitialisée\n")
        else:
            print("✓ Ajout aux documents existants\n")
    
    # Indexer les chunks
    print("4. Indexation des SummarizedChunk...")
    count = store.index_summarized_chunks(
        summarized_chunks,
        batch_size=32
    )
    print(f"✓ {count} documents indexés\n")
    
    # Afficher les statistiques après indexation
    print("5. Statistiques après indexation:")
    stats_after = store.get_collection_stats()
    for key, value in stats_after.items():
        print(f"  {key}: {value}")
    print()
    
    # Test de recherche
    print("6. Test de recherche sémantique:")
    test_queries = [
        "droit",
        "jugement",
        "tribunal",
    ]
    
    for query in test_queries:
        print(f"\n  Requête: '{query}'")
        results = store.search(query, n_results=3)
        
        if results:
            for i, result in enumerate(results, 1):
                print(f"    {i}. [{result['metadata']['spider']}] {result['document'][:100]}...")
                print(f"       Distance: {result['distance']:.4f}" if result['distance'] else "")
        else:
            print("    Aucun résultat")
    
    print("\n✓ Indexation terminée avec succès!")


if __name__ == "__main__":
    main()


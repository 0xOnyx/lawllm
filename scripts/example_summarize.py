"""
Exemple d'utilisation du module de résumé.

Ce script montre comment :
1. Charger des chunks existants
2. Les résumer via un LLM local (Ollama)
3. Sauvegarder les résumés
"""
import sys
import json
from pathlib import Path

from src.rag.summarizer import DEFAULT_MODEL

# Ajouter le répertoire racine du projet au PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src import (
    chunk_document_by_pages,
    ScrapedDocument,
    TextChunk,
    SummarizedChunk,
    summarize_chunks,
    check_ollama_connection,
    list_available_models,
)


def load_chunks(chunks_dir: str = "data/chunks") -> list[TextChunk]:
    """
    Charge les chunks depuis les fichiers JSON.
    
    Args:
        chunks_dir: Répertoire contenant les fichiers de chunks
        
    Returns:
        Liste de TextChunk
    """
    chunks_path = Path(chunks_dir)
    if not chunks_path.exists():
        print(f"Le répertoire {chunks_dir} n'existe pas.")
        return []
    
    all_chunks = []
    
    # Charger tous les fichiers JSON de chunks
    json_files = list(chunks_path.glob("*_chunks.json"))
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
                for chunk_data in chunks_data:
                    chunk = TextChunk(**chunk_data)
                    all_chunks.append(chunk)
        except Exception as e:
            print(f"  Erreur lors du chargement de {json_file}: {e}")
    
    return all_chunks


def save_summarized_chunks(summarized_chunks: list[SummarizedChunk], output_dir: str = "data/summaries"):
    """
    Sauvegarde les SummarizedChunk avec toutes leurs métadonnées.
    
    Args:
        summarized_chunks: Liste des SummarizedChunk à sauvegarder
        output_dir: Répertoire de sortie
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Grouper par document
    chunks_by_doc = {}
    for summarized_chunk in summarized_chunks:
        doc_id = summarized_chunk.document_id
        if doc_id not in chunks_by_doc:
            chunks_by_doc[doc_id] = []
        chunks_by_doc[doc_id].append(summarized_chunk)
    
    # Sauvegarder chaque document
    for doc_id, doc_chunks in chunks_by_doc.items():
        filepath = output_path / f"{doc_id}_summarized.json"
        
        # Convertir en dictionnaires pour la sérialisation JSON
        chunks_data = [chunk.model_dump() for chunk in doc_chunks]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
        
        print(f"  ✓ {len(doc_chunks)} résumés sauvegardés pour {doc_id}")


def main():
    """Pipeline de résumé des chunks."""
    print("=== LawLLM - Résumé des chunks ===\n")
    
    # Vérifier la connexion Ollama
    print("1. Vérification de la connexion Ollama...")
    if not check_ollama_connection():
        print("✗ Ollama n'est pas accessible.")
        print("  Assurez-vous qu'Ollama est démarré: ollama serve")
        print("  Ou installez Ollama depuis: https://ollama.ai")
        return
    
    print("✓ Ollama est accessible\n")
    
    # Lister les modèles disponibles
    print("2. Modèles disponibles dans Ollama:")
    try:
        models = list_available_models()
        for model in models:
            print(f"  - {model}")
        if not models:
            print("  Aucun modèle trouvé. Utilisez 'ollama pull llama3' pour télécharger un modèle.")
            return
    except Exception as e:
        print(f"✗ Erreur lors de la récupération des modèles: {e}")
        return
    
    # Sélectionner le modèle (par défaut: llama3)
    model = DEFAULT_MODEL
    if model not in models:
        print(f"\n⚠ Modèle '{model}' non trouvé. Utilisation du premier modèle disponible.")
        model = models[0]
    
    print(f"\n✓ Utilisation du modèle: {model}\n")
    
    # Charger les chunks
    print("3. Chargement des chunks...")
    chunks = load_chunks("data/chunks")
    
    if not chunks:
        print("Aucun chunk trouvé dans data/chunks")
        print("Utilisez d'abord chunk_documents.py pour créer des chunks.")
        return
    
    print(f"✓ {len(chunks)} chunks chargés\n")
    
    # Limiter le nombre de chunks pour l'exemple (optionnel)
    max_chunks = 1  # Modifier selon vos besoins
    if len(chunks) > max_chunks:
        print(f"⚠ Limitation à {max_chunks} chunks pour l'exemple\n")
        chunks = chunks[:max_chunks]
    
    # Résumer les chunks
    print("4. Résumé des chunks (cela peut prendre du temps)...")
    summarized_chunks = summarize_chunks(chunks, model=model, max_length=500)
    
    print(f"\n✓ {len(summarized_chunks)} résumés créés\n")
    
    # Afficher quelques exemples
    print("5. Exemples de résumés:")
    for i, summarized_chunk in enumerate(summarized_chunks[:3]):
        print(f"\n  Chunk {i+1} ({summarized_chunk.id}):")
        print(f"    Document: {summarized_chunk.document_id}")
        print(f"    Spider: {summarized_chunk.spider}")
        print(f"    Longueur original: {len(summarized_chunk.original_text)} caractères")
        print(f"    Longueur résumé: {len(summarized_chunk.summary)} caractères")
        print(f"    Résumé: {summarized_chunk.summary[:200]}...")
        print(f"    Métadonnées ChromaDB: {list(summarized_chunk.to_chromadb_metadata().keys())}")
    
    # Sauvegarder les résumés
    print("\n6. Sauvegarde des résumés...")
    save_summarized_chunks(summarized_chunks, "data/summaries")
    print("✓ Résumés sauvegardés dans data/summaries/")


if __name__ == "__main__":
    main()


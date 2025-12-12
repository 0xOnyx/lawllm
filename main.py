"""
LawLLM - Script principal avec CLI pour gérer la pipeline complète.

Ce script permet d'ajouter des documents dans ChromaDB via différentes méthodes :
- Scraping depuis l'API (par défaut) -> chunking -> résumé -> indexation
- Depuis des documents scrapés (data/raw)
- Depuis des chunks existants (data/chunks)
- Depuis des résumés existants (data/summaries)
"""
import sys
import json
import argparse
import asyncio
from pathlib import Path
from typing import List, Optional, Set

# Ajouter le répertoire racine au PYTHONPATH
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src import (
    ScrapedDocument,
    TextChunk,
    SummarizedChunk,
    chunk_document_by_pages,
    chunk_document_by_tokens,
    chunk_documents_batch,
    summarize_chunks,
    summarize_chunk,
    check_ollama_connection,
    list_available_models,
    load_summarized_chunks,
    VectorStore,
    EntscheidsucheScraper,
    ScraperConfig,
    list_available_spiders,
)
from src.rag.vector_store import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_DB_PATH,
)
from src.rag.summarizer import DEFAULT_MODEL as DEFAULT_OLLAMA_MODEL


def get_existing_document_ids(data_dir: str = "data/raw") -> Set[str]:
    """
    Récupère l'ensemble des IDs de documents déjà présents dans data_dir.

    Args:
        data_dir: Répertoire contenant les documents JSON

    Returns:
        Ensemble des IDs de documents existants
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        return set()

    existing_ids = set()

    # Parcourir tous les sous-répertoires (spiders)
    for spider_dir in data_path.iterdir():
        if not spider_dir.is_dir():
            continue

        # Charger tous les fichiers JSON
        json_files = list(spider_dir.glob("*.json"))
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'id' in data:
                        existing_ids.add(data['id'])
            except Exception:
                # Ignorer les erreurs de lecture
                pass

    return existing_ids


def load_scraped_documents(data_dir: str = "data/raw") -> List[ScrapedDocument]:
    """
    Charge tous les documents scrapés depuis un répertoire.

    Args:
        data_dir: Répertoire contenant les documents JSON

    Returns:
        Liste de ScrapedDocument
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"✗ Le répertoire {data_dir} n'existe pas.")
        return []

    documents = []

    # Parcourir tous les sous-répertoires (spiders)
    for spider_dir in data_path.iterdir():
        if not spider_dir.is_dir():
            continue

        spider = spider_dir.name
        print(f"  Chargement des documents de {spider}...")

        # Charger tous les fichiers JSON
        json_files = list(spider_dir.glob("*.json"))
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    doc = ScrapedDocument(**data)
                    documents.append(doc)
            except Exception as e:
                print(f"    ✗ Erreur lors du chargement de {json_file}: {e}")

        count = len([d for d in documents if d.spider == spider])
        print(f"    ✓ {count} documents chargés")

    return documents


def load_text_chunks(chunks_dir: str = "data/chunks") -> List[TextChunk]:
    """
    Charge les chunks depuis les fichiers JSON.

    Args:
        chunks_dir: Répertoire contenant les fichiers de chunks

    Returns:
        Liste de TextChunk
    """
    chunks_path = Path(chunks_dir)
    if not chunks_path.exists():
        print(f"✗ Le répertoire {chunks_dir} n'existe pas.")
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
            print(f"  ✗ Erreur lors du chargement de {json_file}: {e}")

    return all_chunks


def load_summarized_chunks_from_dir(summaries_dir: str = "data/summaries") -> List[SummarizedChunk]:
    """
    Charge tous les SummarizedChunk depuis un répertoire.

    Args:
        summaries_dir: Répertoire contenant les fichiers JSON de résumés

    Returns:
        Liste de SummarizedChunk
    """
    summaries_path = Path(summaries_dir)
    if not summaries_path.exists():
        print(f"✗ Le répertoire {summaries_dir} n'existe pas.")
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


def save_chunks(chunks: List[TextChunk], output_dir: str = "data/chunks", append: bool = False):
    """
    Sauvegarde les chunks dans des fichiers JSON.

    Args:
        chunks: Liste de TextChunk
        output_dir: Répertoire de sortie
        append: Si True, ajoute aux chunks existants au lieu de remplacer
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

        if append and filepath.exists():
            # Charger les chunks existants
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    existing_chunks = [TextChunk(**item) for item in existing_data]
                    # Fusionner en évitant les doublons
                    existing_ids = {c.id for c in existing_chunks}
                    new_chunks = [c for c in doc_chunks if c.id not in existing_ids]
                    doc_chunks = existing_chunks + new_chunks
            except Exception:
                pass  # En cas d'erreur, on remplace

        chunks_data = [chunk.model_dump() for chunk in doc_chunks]

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)


def save_summarized_chunks(summarized_chunks: List[SummarizedChunk], output_dir: str = "data/summaries",
                           append: bool = False, show_progress: bool = True):
    """
    Sauvegarde les SummarizedChunk avec toutes leurs métadonnées.

    Args:
        summarized_chunks: Liste des SummarizedChunk à sauvegarder
        output_dir: Répertoire de sortie
        append: Si True, ajoute aux résumés existants au lieu de remplacer
        show_progress: Afficher une barre de progression
    """
    if not summarized_chunks:
        return
        
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Grouper par document
    chunks_by_doc = {}
    for summarized_chunk in summarized_chunks:
        doc_id = summarized_chunk.document_id
        if doc_id not in chunks_by_doc:
            chunks_by_doc[doc_id] = []
        chunks_by_doc[doc_id].append(summarized_chunk)

    # Sauvegarder chaque document avec barre de progression
    from tqdm import tqdm
    
    items = list(chunks_by_doc.items())
    iterator = tqdm(items, desc="Sauvegarde des résumés", unit="doc") if show_progress else items
    
    for doc_id, doc_chunks in iterator:
        filepath = output_path / f"{doc_id}_summarized.json"

        if append and filepath.exists():
            # Charger les résumés existants
            try:
                existing_chunks = load_summarized_chunks(str(filepath))
                # Fusionner en évitant les doublons
                existing_ids = {c.id for c in existing_chunks}
                new_chunks = [c for c in doc_chunks if c.id not in existing_ids]
                doc_chunks = existing_chunks + new_chunks
            except Exception:
                pass  # En cas d'erreur, on remplace

        # Convertir en dictionnaires pour la sérialisation JSON
        chunks_data = [chunk.model_dump() for chunk in doc_chunks]

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)


def summarize_chunks_batch(
        chunks: List[TextChunk],
        model: str = DEFAULT_OLLAMA_MODEL,
        ollama_url: str = "http://localhost:11434",
        max_length: int = 500,
        save_intermediate: bool = True
) -> List[SummarizedChunk]:
    """
    Résume plusieurs chunks.

    Args:
        chunks: Liste des chunks à résumer
        model: Nom du modèle Ollama à utiliser
        ollama_url: URL de l'API Ollama
        max_length: Longueur maximale de chaque résumé en caractères
        save_intermediate: Si True, sauvegarde les résumés après chaque chunk

    Returns:
        Liste des SummarizedChunk
    """
    from tqdm import tqdm
    
    summarized_chunks = []
    
    # Sauvegarder par batch pour éviter trop d'écritures disque
    batch_to_save = []
    batch_size = 100  # Sauvegarder tous les 100 chunks
    
    with tqdm(total=len(chunks), desc="Résumé des chunks", unit="chunk") as pbar:
        for chunk in chunks:
            try:
                summarized_chunk = summarize_chunk(chunk, model, ollama_url, max_length)
                summarized_chunks.append(summarized_chunk)
                
                # Ajouter au batch à sauvegarder
                if save_intermediate:
                    batch_to_save.append(summarized_chunk)
                    
                    # Sauvegarder par batch
                    if len(batch_to_save) >= batch_size:
                        save_summarized_chunks(batch_to_save, "data/summaries", append=True, show_progress=False)
                        batch_to_save = []
                        
            except Exception as e:
                print(f"\n✗ Erreur lors du résumé du chunk {chunk.id}: {e}")
                # Fallback
                fallback_summary = chunk.text
                if len(chunk.text) > max_length:
                    fallback_summary += "..."
                fallback_chunk = SummarizedChunk.from_text_chunk(chunk, fallback_summary)
                summarized_chunks.append(fallback_chunk)
                
                if save_intermediate:
                    batch_to_save.append(fallback_chunk)
                    if len(batch_to_save) >= batch_size:
                        save_summarized_chunks(batch_to_save, "data/summaries", append=True, show_progress=False)
                        batch_to_save = []
            
            pbar.update(1)
        
        # Sauvegarder les résumés restants
        if save_intermediate and batch_to_save:
            save_summarized_chunks(batch_to_save, "data/summaries", append=True, show_progress=False)

    return summarized_chunks


def pipeline_from_documents(
        documents: List[ScrapedDocument],
        store: VectorStore,
        ollama_model: str = DEFAULT_OLLAMA_MODEL,
        ollama_url: str = "http://localhost:11434",
        save_intermediate: bool = True,
        skip_summarization: bool = False,
        max_tokens: int = 500,
        max_workers: Optional[int] = None,
        embedding_batch_size: Optional[int] = None
) -> int:
    """
    Pipeline complète : documents -> chunks -> résumés -> indexation.

    Args:
        documents: Liste de documents à traiter
        store: Instance de VectorStore pour l'indexation
        ollama_model: Modèle Ollama à utiliser pour le résumé
        ollama_url: URL de l'API Ollama
        save_intermediate: Si True, sauvegarde les chunks et résumés
        skip_summarization: Si True, utilise le texte original au lieu de résumer
        max_tokens: Nombre maximum de tokens par chunk (utilisé avec skip_summarization)
        max_workers: Nombre de processus parallèles pour le chunking
        embedding_batch_size: Taille des batches pour les embeddings (None = auto)

    Returns:
        Nombre de documents indexés
    """
    all_chunks = []

    if skip_summarization:
        overlap_tokens = int(max_tokens * 0.2)
        print(f"\n=== Étape 1: Découpage en chunks par tokens (max {max_tokens} tokens, overlap ~{overlap_tokens} tokens) ===")

        # Utiliser la fonction batch parallélisée avec barre de progression
        all_chunks = chunk_documents_batch(
            documents,
            max_tokens=max_tokens,
            overlap_ratio=0.2,
            max_workers=max_workers,
            desc="Création des chunks"
        )

        # Sauvegarder tous les chunks en une seule fois (plus efficace)
        if save_intermediate and all_chunks:
            save_chunks(all_chunks, "data/chunks", append=True)

        print(f"\n✓ {len(all_chunks)} chunks créés au total")

        # Créer des SummarizedChunk avec le texte complet (pas de résumé)
        print("\n=== Étape 2: Création de SummarizedChunk (texte complet, pas de résumé) ===")
        summarized_chunks = []
        for chunk in all_chunks:
            # Utiliser le texte complet comme "résumé" (car on ne résume pas)
            summarized_chunk = SummarizedChunk.from_text_chunk(chunk, chunk.text)
            summarized_chunks.append(summarized_chunk)
    else:
        print("\n=== Étape 1: Découpage en chunks ===")
        
        # Utiliser une approche parallèle pour le chunking par pages aussi
        # On peut paralléliser avec ProcessPoolExecutor
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from tqdm import tqdm
        
        def _chunk_pages_worker(doc_dict):
            """Worker pour chunker par pages."""
            doc = ScrapedDocument(**doc_dict)
            return chunk_document_by_pages(doc, pages_per_chunk=3)
        
        doc_dicts = [doc.model_dump() for doc in documents]
        all_chunks = []
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_doc = {
                executor.submit(_chunk_pages_worker, doc_dict): i
                for i, doc_dict in enumerate(doc_dicts)
            }
            
            with tqdm(total=len(documents), desc="Création des chunks", unit="doc") as pbar:
                for future in as_completed(future_to_doc):
                    try:
                        chunks = future.result()
                        all_chunks.extend(chunks)
                    except Exception as e:
                        doc_idx = future_to_doc[future]
                        doc_id = documents[doc_idx].id if doc_idx < len(documents) else "unknown"
                        print(f"\n⚠ Erreur lors du chunking de {doc_id}: {e}")
                    finally:
                        pbar.update(1)
        
        # Sauvegarder tous les chunks en une seule fois
        if save_intermediate and all_chunks:
            save_chunks(all_chunks, "data/chunks", append=True)

        print(f"\n✓ {len(all_chunks)} chunks créés au total")

        print("\n=== Étape 2: Résumé des chunks ===")
        # Vérifier la connexion Ollama
        if not check_ollama_connection(ollama_url):
            print(f"✗ Ollama n'est pas accessible à {ollama_url}")
            print("  Utilisation du texte original tronqué comme résumé...")
            summarized_chunks = []
            for chunk in all_chunks:
                summary = chunk.text
                summarized_chunk = SummarizedChunk.from_text_chunk(chunk, summary)
                summarized_chunks.append(summarized_chunk)
            
            # Sauvegarder tous les résumés en une fois
            if save_intermediate and summarized_chunks:
                save_summarized_chunks(summarized_chunks, "data/summaries", append=True)
        else:
            print(f"  Utilisation du modèle: {ollama_model}")
            summarized_chunks = summarize_chunks_batch(
                all_chunks,
                model=ollama_model,
                ollama_url=ollama_url,
                max_length=500,
                save_intermediate=save_intermediate
            )

    print(f"\n✓ {len(summarized_chunks)} résumés créés")

    print("\n=== Étape 3: Indexation dans ChromaDB ===")
    count = store.index_summarized_chunks(
        summarized_chunks, 
        batch_size=128,
        embedding_batch_size=embedding_batch_size
    )
    print(f"\n✓ {count} documents indexés dans ChromaDB")

    return count


def pipeline_from_chunks(
        chunks: List[TextChunk],
        store: VectorStore,
        ollama_model: str = DEFAULT_OLLAMA_MODEL,
        ollama_url: str = "http://localhost:11434",
        save_intermediate: bool = True,
        skip_summarization: bool = False
) -> int:
    """
    Pipeline : chunks -> résumés -> indexation.

    Args:
        chunks: Liste de chunks à traiter
        store: Instance de VectorStore pour l'indexation
        ollama_model: Modèle Ollama à utiliser pour le résumé
        ollama_url: URL de l'API Ollama
        save_intermediate: Si True, sauvegarde les résumés
        skip_summarization: Si True, utilise le texte original au lieu de résumer

    Returns:
        Nombre de documents indexés
    """
    if skip_summarization:
        print("\n=== Création de SummarizedChunk (sans résumé) ===")
        summarized_chunks = []
        for chunk in chunks:
            summary = chunk.text
            summarized_chunk = SummarizedChunk.from_text_chunk(chunk, summary)
            summarized_chunks.append(summarized_chunk)
        
        # Sauvegarder tous les résumés en une fois
        if save_intermediate and summarized_chunks:
            save_summarized_chunks(summarized_chunks, "data/summaries", append=True)
    else:
        print("\n=== Résumé des chunks ===")
        # Vérifier la connexion Ollama
        if not check_ollama_connection(ollama_url):
            print(f"✗ Ollama n'est pas accessible à {ollama_url}")
            print("  Utilisation du texte original tronqué comme résumé...")
            summarized_chunks = []
            for chunk in chunks:
                summary = chunk.text
                summarized_chunk = SummarizedChunk.from_text_chunk(chunk, summary)
                summarized_chunks.append(summarized_chunk)
            
            # Sauvegarder tous les résumés en une fois
            if save_intermediate and summarized_chunks:
                save_summarized_chunks(summarized_chunks, "data/summaries", append=True)
        else:
            print(f"  Utilisation du modèle: {ollama_model}")
            summarized_chunks = summarize_chunks_batch(
                chunks,
                model=ollama_model,
                ollama_url=ollama_url,
                max_length=500,
                save_intermediate=save_intermediate
            )

    print(f"\n✓ {len(summarized_chunks)} résumés créés")

    print("\n=== Indexation dans ChromaDB ===")
    count = store.index_summarized_chunks(summarized_chunks, batch_size=32)
    print(f"\n✓ {count} documents indexés dans ChromaDB")

    return count


async def list_spiders_with_counts(
        rate_limit: int = 5
) -> List[tuple]:
    """
    Liste tous les spiders disponibles avec le nombre de documents pour chacun.

    Args:
        rate_limit: Limite de requêtes par seconde

    Returns:
        Liste de tuples (spider_id, description, document_count)
    """
    print("=== Récupération de la liste des spiders et du nombre de documents ===\n")

    # Obtenir la liste des spiders disponibles
    available_spiders = list_available_spiders()

    if not available_spiders:
        print("✗ Aucun spider disponible.")
        return []

    # Configuration du scraper
    config = ScraperConfig(rate_limit=rate_limit)

    spider_info = []

    async with EntscheidsucheScraper(config) as scraper:
        print(f"Récupération du nombre de documents pour {len(available_spiders)} spiders...\n")

        for i, (spider_id, description) in enumerate(available_spiders.items(), 1):
            try:
                # Récupérer la liste des documents pour compter
                json_paths = await scraper._client.list_documents(spider_id)
                doc_count = len(json_paths) if json_paths else 0
                spider_info.append((spider_id, description, doc_count))

                # Afficher la progression
                if i % 10 == 0 or i == len(available_spiders):
                    print(f"  Progression: {i}/{len(available_spiders)} spiders traités...")
            except Exception as e:
                # En cas d'erreur, mettre 0 documents
                spider_info.append((spider_id, description, 0))
                print(f"  ⚠ Erreur pour {spider_id}: {e}")

    return spider_info


async def scrape_documents(
        spiders: Optional[List[str]] = None,
        max_docs_per_spider: Optional[int] = None,
        language: Optional[str] = None,
        only_new: bool = False,
        output_dir: str = "data/raw",
        rate_limit: int = 5,
        max_concurrent: int = 10
) -> List[ScrapedDocument]:
    """
    Scrape des documents depuis l'API Entscheidsuche.

    Args:
        spiders: Liste des spiders à scraper (None = tous les spiders disponibles)
        max_docs_per_spider: Nombre maximum de documents par spider
        language: Filtrer par langue (de, fr, it)
        only_new: Si True, ne scraper que les documents qui n'existent pas déjà
        output_dir: Répertoire de sortie
        rate_limit: Limite de requêtes par seconde
        max_concurrent: Nombre maximum de requêtes simultanées

    Returns:
        Liste des documents scrapés
    """
    print("=== Scraping depuis l'API Entscheidsuche ===\n")

    # Obtenir les documents existants si on ne veut que les nouveaux
    existing_ids = set()
    if only_new:
        print("Vérification des documents existants...")
        existing_ids = get_existing_document_ids(output_dir)
        print(f"  {len(existing_ids)} documents déjà présents")
        if existing_ids:
            print("  (Seuls les nouveaux documents seront téléchargés)\n")

    # Obtenir la liste des spiders disponibles
    available_spiders = list_available_spiders()

    # Déterminer quels spiders scraper
    if spiders is None:
        # Si aucun spider spécifié, utiliser tous les spiders disponibles
        spiders = list(available_spiders.keys())
        print(f"Scraping de tous les spiders disponibles ({len(spiders)} spiders)\n")
    else:
        # Vérifier que les spiders demandés existent
        invalid_spiders = [s for s in spiders if s not in available_spiders]
        if invalid_spiders:
            print(f"⚠ Spiders invalides ignorés: {', '.join(invalid_spiders)}")
        spiders = [s for s in spiders if s in available_spiders]

        if not spiders:
            print("✗ Aucun spider valide à scraper.")
            return []

        print(f"Scraping de {len(spiders)} spider(s): {', '.join(spiders)}\n")

    # Configuration du scraper
    config = ScraperConfig(
        output_dir=output_dir,
        rate_limit=rate_limit,
        max_concurrent=max_concurrent
    )

    documents = []
    new_count = 0
    skipped_count = 0
    processed_doc_ids = set(existing_ids) if existing_ids else set()

    async with EntscheidsucheScraper(config) as scraper:
        for spider in spiders:
            description = available_spiders.get(spider, 'Inconnu')
            print(f"\n=== {spider} ({description}) ===")

            spider_docs = []
            try:
                async for doc in scraper.fetch_spider(spider, max_docs_per_spider, language):
                    # Vérifier si le document existe déjà
                    if doc.id in processed_doc_ids:
                        skipped_count += 1
                        continue

                    # Sauvegarder le document
                    await scraper.save_document(doc)
                    documents.append(doc)
                    spider_docs.append(doc)
                    new_count += 1
                    processed_doc_ids.add(doc.id)
            except Exception as e:
                print(f"  ✗ Erreur lors du scraping de {spider}: {e}")

            print(f"  ✓ {len(spider_docs)} nouveaux documents scrapés pour {spider}")

    print(f"\n✓ Scraping terminé: {new_count} nouveaux documents, {skipped_count} documents ignorés (déjà présents)")

    return documents


def main():
    """Fonction principale avec CLI."""
    parser = argparse.ArgumentParser(
        description="LawLLM - Pipeline complète pour ajouter des documents dans ChromaDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  # Pipeline complète avec scraping (par défaut)
  python main.py

  # Scraper uniquement certains spiders
  python main.py --spiders CH_BGer VD_FindInfo

  # Scraper uniquement les nouveaux documents
  python main.py --only-new

  # Scraper avec limite de documents par spider
  python main.py --spiders CH_BGer --max-docs 50

  # Indexer depuis des résumés existants (sans scraping)
  python main.py --from-summaries

  # Indexer depuis des chunks (avec résumé)
  python main.py --from-chunks

  # Pipeline depuis des documents déjà scrapés
  python main.py --from-documents

  # Utiliser un modèle Ollama spécifique
  python main.py --ollama-model llama3

  # Lister tous les spiders disponibles avec le nombre de documents
  python main.py --list-spiders

  # Accélérer le téléchargement (plus de requêtes simultanées)
  python main.py --rate-limit 10 --max-concurrent 20
        """
    )

    # Option pour lister les spiders
    parser.add_argument(
        "--list-spiders",
        action="store_true",
        help="Afficher la liste des spiders disponibles avec le nombre de documents"
    )

    # Source des données
    source_group = parser.add_mutually_exclusive_group(required=False)
    source_group.add_argument(
        "--from-summaries",
        action="store_true",
        help="Charger depuis data/summaries (résumés déjà créés, pas de scraping)"
    )
    source_group.add_argument(
        "--from-chunks",
        action="store_true",
        help="Charger depuis data/chunks (chunks déjà créés, pas de scraping)"
    )
    source_group.add_argument(
        "--from-documents",
        action="store_true",
        help="Charger depuis data/raw (documents déjà scrapés, pas de scraping)"
    )

    # Options de scraping
    parser.add_argument(
        "--spiders",
        nargs="+",
        type=str,
        help="Liste des spiders à scraper (ex: CH_BGer VD_FindInfo). Par défaut: tous les spiders"
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        help="Nombre maximum de documents à scraper par spider"
    )
    parser.add_argument(
        "--language",
        type=str,
        choices=["de", "fr", "it"],
        help="Filtrer par langue (de, fr, it)"
    )
    parser.add_argument(
        "--only-new",
        action="store_true",
        help="Ne scraper que les documents qui n'existent pas déjà dans data/raw"
    )
    parser.add_argument(
        "--rate-limit",
        type=int,
        default=5,
        help="Limite de requêtes par seconde pour le scraping (défaut: 5). Augmentez pour accélérer (ex: 10-20)"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Nombre maximum de requêtes simultanées (défaut: 10). Augmentez pour accélérer (ex: 20-50)"
    )

    # Options ChromaDB
    parser.add_argument(
        "--db-path",
        type=str,
        default=DEFAULT_DB_PATH,
        help=f"Chemin vers la base ChromaDB (défaut: {DEFAULT_DB_PATH})"
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default=DEFAULT_COLLECTION_NAME,
        help=f"Nom de la collection ChromaDB (défaut: {DEFAULT_COLLECTION_NAME})"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"Modèle d'embedding (défaut: {DEFAULT_EMBEDDING_MODEL})"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=['cpu', 'cuda'],
        help="Device pour le modèle d'embedding (None = auto-détection GPU, 'cpu' ou 'cuda'). Force l'utilisation du GPU si 'cuda'"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Réinitialiser la collection avant l'indexation"
    )

    # Options Ollama (pour le résumé)
    parser.add_argument(
        "--ollama-model",
        type=str,
        default=DEFAULT_OLLAMA_MODEL,
        help=f"Modèle Ollama pour le résumé (défaut: {DEFAULT_OLLAMA_MODEL})"
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434",
        help="URL de l'API Ollama (défaut: http://localhost:11434)"
    )
    parser.add_argument(
        "--skip-summarization",
        action="store_true",
        help="Ne pas résumer. Crée des chunks par tokens avec chevauchement au lieu de résumer"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=500,
        help="Nombre maximum de tokens par chunk (défaut: 500). Utilisé avec --skip-summarization. L'overlap sera calculé automatiquement (20%% du max_tokens)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Nombre maximum de processus parallèles pour le chunking (défaut: nombre de CPUs). Augmentez pour accélérer le traitement de gros volumes"
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=None,
        help="Taille des batches pour la génération d'embeddings (défaut: auto selon le volume). Augmentez (ex: 256-512) pour accélérer avec GPU"
    )

    # Options de sauvegarde
    parser.add_argument(
        "--no-save-intermediate",
        action="store_true",
        help="Ne pas sauvegarder les chunks/résumés intermédiaires"
    )

    # Options de répertoires
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Répertoire des documents scrapés (défaut: data/raw)"
    )
    parser.add_argument(
        "--chunks-dir",
        type=str,
        default="data/chunks",
        help="Répertoire des chunks (défaut: data/chunks)"
    )
    parser.add_argument(
        "--summaries-dir",
        type=str,
        default="data/summaries",
        help="Répertoire des résumés (défaut: data/summaries)"
    )

    args = parser.parse_args()

    # Si l'option --list-spiders est activée, afficher la liste et quitter
    if args.list_spiders:
        print("=" * 80)
        print("LawLLM - Liste des spiders disponibles")
        print("=" * 80)
        print()

        spider_info = asyncio.run(list_spiders_with_counts(rate_limit=args.rate_limit))

        if not spider_info:
            print("✗ Aucun spider trouvé.")
            return 1

        # Trier par nombre de documents (décroissant)
        spider_info.sort(key=lambda x: x[2], reverse=True)

        # Afficher les résultats
        print("\n" + "=" * 80)
        print(f"{'Spider ID':<20} {'Description':<50} {'Documents':>10}")
        print("=" * 80)

        total_docs = 0
        for spider_id, description, doc_count in spider_info:
            # Tronquer la description si trop longue
            desc_display = description[:47] + "..." if len(description) > 50 else description
            print(f"{spider_id:<20} {desc_display:<50} {doc_count:>10,}")
            total_docs += doc_count

        print("=" * 80)
        print(f"{'TOTAL':<20} {'':<50} {total_docs:>10,}")
        print("=" * 80)
        print()

        # Afficher quelques statistiques
        spiders_with_docs = [s for s in spider_info if s[2] > 0]
        spiders_without_docs = [s for s in spider_info if s[2] == 0]

        print(f"Statistiques:")
        print(f"  - Spiders avec documents: {len(spiders_with_docs)}")
        print(f"  - Spiders sans documents: {len(spiders_without_docs)}")
        print(f"  - Total de documents: {total_docs:,}")

        if spiders_with_docs:
            max_spider = max(spiders_with_docs, key=lambda x: x[2])
            print(f"  - Spider avec le plus de documents: {max_spider[0]} ({max_spider[2]:,} documents)")

        return 0

    # Afficher le header
    print("=" * 60)
    print("LawLLM - Pipeline d'indexation dans ChromaDB")
    print("=" * 60)
    print()

    # Initialiser le VectorStore
    print("=== Initialisation du VectorStore ===")
    store = VectorStore(
        collection_name=args.collection_name,
        db_path=args.db_path,
        embedding_model=args.embedding_model,
        device=args.device
    )
    print()

    # Afficher les statistiques avant
    stats_before = store.get_collection_stats()
    print(f"Documents existants: {stats_before['total_documents']}")

    # Réinitialiser si demandé
    if args.reset:
        print("\n⚠ Réinitialisation de la collection...")
        store.reset_collection()
        print("✓ Collection réinitialisée\n")
    elif stats_before['total_documents'] > 0:
        print("  (Les nouveaux documents seront ajoutés aux existants)")
    print()

    # Traiter selon la source
    count = 0

    # Si aucune source spécifiée, faire du scraping par défaut
    if not args.from_summaries and not args.from_chunks and not args.from_documents:
        # Pipeline complète avec scraping
        print("=== Mode par défaut: Scraping -> Chunking -> Résumé -> Indexation ===\n")

        # Scraper les documents
        documents = asyncio.run(scrape_documents(
            spiders=args.spiders,
            max_docs_per_spider=args.max_docs,
            language=args.language,
            only_new=args.only_new,
            output_dir=args.data_dir,
            rate_limit=args.rate_limit,
            max_concurrent=args.max_concurrent
        ))

        if not documents:
            print("✗ Aucun document scrapé.")
            return 1

        print(f"\n✓ {len(documents)} documents scrapés\n")

        # Continuer avec la pipeline
        count = pipeline_from_documents(
            documents,
            store,
            ollama_model=args.ollama_model,
            ollama_url=args.ollama_url,
            save_intermediate=not args.no_save_intermediate,
            skip_summarization=args.skip_summarization,
            max_tokens=args.max_tokens,
            max_workers=args.max_workers,
            embedding_batch_size=args.embedding_batch_size
        )

    elif args.from_summaries:
        print("=== Chargement depuis les résumés ===")
        summarized_chunks = load_summarized_chunks_from_dir(args.summaries_dir)

        if not summarized_chunks:
            print("✗ Aucun résumé trouvé.")
            return 1

        print(f"\n✓ {len(summarized_chunks)} résumés chargés\n")

        print("=== Indexation dans ChromaDB ===")
        count = store.index_summarized_chunks(summarized_chunks, batch_size=32)

    elif args.from_chunks:
        print("=== Chargement depuis les chunks ===")
        chunks = load_text_chunks(args.chunks_dir)

        if not chunks:
            print("✗ Aucun chunk trouvé.")
            return 1

        print(f"\n✓ {len(chunks)} chunks chargés\n")

        count = pipeline_from_chunks(
            chunks,
            store,
            ollama_model=args.ollama_model,
            ollama_url=args.ollama_url,
            save_intermediate=not args.no_save_intermediate,
            skip_summarization=args.skip_summarization
        )

    elif args.from_documents:
        print("=== Chargement depuis les documents ===")
        documents = load_scraped_documents(args.data_dir)

        if not documents:
            print("✗ Aucun document trouvé.")
            return 1

        print(f"\n✓ {len(documents)} documents chargés\n")

        count = pipeline_from_documents(
            documents,
            store,
            ollama_model=args.ollama_model,
            ollama_url=args.ollama_url,
            save_intermediate=not args.no_save_intermediate,
            skip_summarization=args.skip_summarization,
            max_tokens=args.max_tokens,
            max_workers=args.max_workers,
            embedding_batch_size=args.embedding_batch_size
        )

    # Afficher les statistiques finales
    print("\n" + "=" * 60)
    print("=== Statistiques finales ===")
    stats_after = store.get_collection_stats()
    for key, value in stats_after.items():
        print(f"  {key}: {value}")
    print()
    print(f"✓ {count} nouveaux documents indexés avec succès!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    # Gérer asyncio sur Windows
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    sys.exit(main())


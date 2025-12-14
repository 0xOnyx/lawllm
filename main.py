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
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import List, Optional, Set, Union
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

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


# =============================================================================
# Configuration
# =============================================================================

class PipelineSource(Enum):
    """Point d'entrée de la pipeline."""
    SCRAPE = auto()      # Scraping -> Documents -> Chunks -> Résumés -> Index
    DOCUMENTS = auto()   # Documents -> Chunks -> Résumés -> Index
    CHUNKS = auto()      # Chunks -> Résumés -> Index
    SUMMARIES = auto()   # Résumés -> Index


@dataclass
class PipelineConfig:
    """Configuration centralisée pour toute la pipeline."""
    
    # Source de données
    source: PipelineSource = PipelineSource.SCRAPE
    
    # Répertoires
    data_dir: str = "data/raw"
    chunks_dir: str = "data/chunks"
    summaries_dir: str = "data/summaries"
    
    # Options de scraping
    spiders: Optional[List[str]] = None
    max_docs_per_spider: Optional[int] = None
    language: Optional[str] = None
    only_new: bool = False
    rate_limit: int = 5
    max_concurrent: int = 10
    
    # Options ChromaDB
    db_path: str = DEFAULT_DB_PATH
    collection_name: str = DEFAULT_COLLECTION_NAME
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    device: Optional[str] = None
    reset_collection: bool = False
    
    # Options Ollama (résumé)
    ollama_model: str = DEFAULT_OLLAMA_MODEL
    ollama_url: str = "http://localhost:11434"
    skip_summarization: bool = False
    max_summary_length: int = 500
    
    # Options de chunking (max_tokens=512 pour respecter la limite du modèle E5)
    max_tokens: int = 512
    overlap_ratio: float = 0.2
    pages_per_chunk: int = 3
    max_workers: Optional[int] = None
    
    # Options d'indexation
    index_batch_size: int = 128
    embedding_batch_size: Optional[int] = None
    
    # Options générales
    save_intermediate: bool = True


# =============================================================================
# Fonctions utilitaires pour le chargement/sauvegarde
# =============================================================================

def get_existing_document_ids(data_dir: str = "data/raw") -> Set[str]:
    """Récupère l'ensemble des IDs de documents déjà présents dans data_dir."""
    data_path = Path(data_dir)
    if not data_path.exists():
        return set()

    existing_ids = set()
    for spider_dir in data_path.iterdir():
        if not spider_dir.is_dir():
            continue
        for json_file in spider_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'id' in data:
                        existing_ids.add(data['id'])
            except Exception:
                pass
    return existing_ids


def load_scraped_documents(data_dir: str = "data/raw") -> List[ScrapedDocument]:
    """Charge tous les documents scrapés depuis un répertoire."""
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"✗ Le répertoire {data_dir} n'existe pas.")
        return []

    documents = []
    for spider_dir in data_path.iterdir():
        if not spider_dir.is_dir():
            continue
        spider = spider_dir.name
        print(f"  Chargement des documents de {spider}...")
        
        for json_file in spider_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    doc = ScrapedDocument(**json.load(f))
                    documents.append(doc)
            except Exception as e:
                print(f"    ✗ Erreur lors du chargement de {json_file}: {e}")
        
        count = len([d for d in documents if d.spider == spider])
        print(f"    ✓ {count} documents chargés")
    return documents


def load_text_chunks(chunks_dir: str = "data/chunks") -> List[TextChunk]:
    """Charge les chunks depuis les fichiers JSON."""
    chunks_path = Path(chunks_dir)
    if not chunks_path.exists():
        print(f"✗ Le répertoire {chunks_dir} n'existe pas.")
        return []

    all_chunks = []
    for json_file in chunks_path.glob("*_chunks.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                for chunk_data in json.load(f):
                    all_chunks.append(TextChunk(**chunk_data))
        except Exception as e:
            print(f"  ✗ Erreur lors du chargement de {json_file}: {e}")
    return all_chunks


def load_summarized_chunks_from_dir(summaries_dir: str = "data/summaries") -> List[SummarizedChunk]:
    """Charge tous les SummarizedChunk depuis un répertoire."""
    summaries_path = Path(summaries_dir)
    if not summaries_path.exists():
        print(f"✗ Le répertoire {summaries_dir} n'existe pas.")
        return []

    all_chunks = []
    for json_file in summaries_path.glob("*_summarized.json"):
        try:
            chunks = load_summarized_chunks(str(json_file))
            all_chunks.extend(chunks)
            print(f"  ✓ {len(chunks)} chunks chargés depuis {json_file.name}")
        except Exception as e:
            print(f"  ✗ Erreur lors du chargement de {json_file}: {e}")
    return all_chunks


def save_chunks(chunks: List[TextChunk], output_dir: str = "data/chunks", append: bool = False):
    """Sauvegarde les chunks dans des fichiers JSON."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Grouper par document
    chunks_by_doc = {}
    for chunk in chunks:
        chunks_by_doc.setdefault(chunk.document_id, []).append(chunk)

    for doc_id, doc_chunks in chunks_by_doc.items():
        filepath = output_path / f"{doc_id}_chunks.json"
        
        if append and filepath.exists():
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    existing = [TextChunk(**item) for item in json.load(f)]
                    existing_ids = {c.id for c in existing}
                    doc_chunks = existing + [c for c in doc_chunks if c.id not in existing_ids]
            except Exception:
                pass

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump([c.model_dump() for c in doc_chunks], f, indent=2, ensure_ascii=False)


def save_summarized_chunks(
    summarized_chunks: List[SummarizedChunk],
    output_dir: str = "data/summaries",
    append: bool = False,
    show_progress: bool = True
):
    """Sauvegarde les SummarizedChunk avec toutes leurs métadonnées."""
    if not summarized_chunks:
        return
        
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Grouper par document
    chunks_by_doc = {}
    for chunk in summarized_chunks:
        chunks_by_doc.setdefault(chunk.document_id, []).append(chunk)

    items = list(chunks_by_doc.items())
    iterator = tqdm(items, desc="Sauvegarde des résumés", unit="doc") if show_progress else items
    
    for doc_id, doc_chunks in iterator:
        filepath = output_path / f"{doc_id}_summarized.json"
        
        if append and filepath.exists():
            try:
                existing = load_summarized_chunks(str(filepath))
                existing_ids = {c.id for c in existing}
                doc_chunks = existing + [c for c in doc_chunks if c.id not in existing_ids]
            except Exception:
                pass

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump([c.model_dump() for c in doc_chunks], f, indent=2, ensure_ascii=False)


# =============================================================================
# Pipeline unifiée
# =============================================================================

class Pipeline:
    """
    Pipeline unifiée pour le traitement des documents juridiques.
    
    Étapes de la pipeline :
    1. Scraping (optionnel) : téléchargement depuis l'API
    2. Chunking : découpage des documents en chunks
    3. Summarization : résumé des chunks (ou passthrough)
    4. Indexation : ajout dans ChromaDB
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self._store: Optional[VectorStore] = None
    
    @property
    def store(self) -> VectorStore:
        """Lazy-load du VectorStore."""
        if self._store is None:
            print("=== Initialisation du VectorStore ===")
            self._store = VectorStore(
                collection_name=self.config.collection_name,
                db_path=self.config.db_path,
                embedding_model=self.config.embedding_model,
                device=self.config.device
            )
            print()
        return self._store
    
    # -------------------------------------------------------------------------
    # Étape 1: Scraping
    # -------------------------------------------------------------------------
    
    async def scrape(self) -> List[ScrapedDocument]:
        """Scrape les documents depuis l'API Entscheidsuche."""
        print("=== Scraping depuis l'API Entscheidsuche ===\n")
        
        existing_ids = set()
        if self.config.only_new:
            print("Vérification des documents existants...")
            existing_ids = get_existing_document_ids(self.config.data_dir)
            print(f"  {len(existing_ids)} documents déjà présents")
            if existing_ids:
                print("  (Seuls les nouveaux documents seront téléchargés)\n")

        available_spiders = list_available_spiders()
        spiders = self.config.spiders
        
        if spiders is None:
            spiders = list(available_spiders.keys())
            print(f"Scraping de tous les spiders disponibles ({len(spiders)} spiders)\n")
        else:
            invalid = [s for s in spiders if s not in available_spiders]
            if invalid:
                print(f"⚠ Spiders invalides ignorés: {', '.join(invalid)}")
            spiders = [s for s in spiders if s in available_spiders]
            if not spiders:
                print("✗ Aucun spider valide à scraper.")
                return []
            print(f"Scraping de {len(spiders)} spider(s): {', '.join(spiders)}\n")

        scraper_config = ScraperConfig(
            output_dir=self.config.data_dir,
            rate_limit=self.config.rate_limit,
            max_concurrent=self.config.max_concurrent
        )

        documents = []
        new_count = 0
        skipped_count = 0
        processed_ids = set(existing_ids)

        async with EntscheidsucheScraper(scraper_config) as scraper:
            for spider in spiders:
                description = available_spiders.get(spider, 'Inconnu')
                print(f"\n=== {spider} ({description}) ===")
                spider_docs = []
                
                try:
                    async for doc in scraper.fetch_spider(
                        spider, 
                        self.config.max_docs_per_spider, 
                        self.config.language
                    ):
                        if doc.id in processed_ids:
                            skipped_count += 1
                            continue
                        
                        await scraper.save_document(doc)
                        documents.append(doc)
                        spider_docs.append(doc)
                        new_count += 1
                        processed_ids.add(doc.id)
                except Exception as e:
                    print(f"  ✗ Erreur lors du scraping de {spider}: {e}")
                
                print(f"  ✓ {len(spider_docs)} nouveaux documents scrapés")

        print(f"\n✓ Scraping terminé: {new_count} nouveaux, {skipped_count} ignorés")
        return documents
    
    # -------------------------------------------------------------------------
    # Étape 2: Chunking
    # -------------------------------------------------------------------------
    
    def chunk_documents(self, documents: List[ScrapedDocument]) -> List[TextChunk]:
        """Découpe les documents en chunks."""
        if self.config.skip_summarization:
            # Chunking par tokens avec overlap
            overlap = int(self.config.max_tokens * self.config.overlap_ratio)
            print(f"\n=== Découpage en chunks par tokens (max {self.config.max_tokens}, overlap ~{overlap}) ===")
            
            chunks = chunk_documents_batch(
                documents,
                max_tokens=self.config.max_tokens,
                overlap_ratio=self.config.overlap_ratio,
                max_workers=self.config.max_workers,
                desc="Création des chunks"
            )
        else:
            # Chunking par pages
            print("\n=== Découpage en chunks par pages ===")
            chunks = self._chunk_by_pages_parallel(documents)
        
        if self.config.save_intermediate and chunks:
            save_chunks(chunks, self.config.chunks_dir, append=True)
        
        print(f"\n✓ {len(chunks)} chunks créés")
        return chunks
    
    def _chunk_by_pages_parallel(self, documents: List[ScrapedDocument]) -> List[TextChunk]:
        """Chunking par pages en parallèle."""
        def worker(doc_dict):
            doc = ScrapedDocument(**doc_dict)
            return chunk_document_by_pages(doc, pages_per_chunk=self.config.pages_per_chunk)
        
        doc_dicts = [doc.model_dump() for doc in documents]
        all_chunks = []
        
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {executor.submit(worker, d): i for i, d in enumerate(doc_dicts)}
            
            with tqdm(total=len(documents), desc="Création des chunks", unit="doc") as pbar:
                for future in as_completed(futures):
                    try:
                        all_chunks.extend(future.result())
                    except Exception as e:
                        idx = futures[future]
                        doc_id = documents[idx].id if idx < len(documents) else "unknown"
                        print(f"\n⚠ Erreur chunking {doc_id}: {e}")
                    finally:
                        pbar.update(1)
        
        return all_chunks
    
    # -------------------------------------------------------------------------
    # Étape 3: Summarization
    # -------------------------------------------------------------------------
    
    def summarize(self, chunks: List[TextChunk]) -> List[SummarizedChunk]:
        """Résume les chunks (ou passthrough si skip_summarization)."""
        if self.config.skip_summarization:
            print("\n=== Création de SummarizedChunk (texte complet, sans résumé) ===")
            return [SummarizedChunk.from_text_chunk(c, c.text) for c in chunks]
        
        print("\n=== Résumé des chunks ===")
        
        if not check_ollama_connection(self.config.ollama_url):
            print(f"✗ Ollama non accessible à {self.config.ollama_url}")
            print("  Utilisation du texte original comme fallback...")
            summarized = [SummarizedChunk.from_text_chunk(c, c.text) for c in chunks]
            
            if self.config.save_intermediate:
                save_summarized_chunks(summarized, self.config.summaries_dir, append=True)
            return summarized
        
        print(f"  Utilisation du modèle: {self.config.ollama_model}")
        return self._summarize_batch(chunks)
    
    def _summarize_batch(self, chunks: List[TextChunk]) -> List[SummarizedChunk]:
        """Résume les chunks par batch avec sauvegarde intermédiaire."""
        summarized = []
        batch_to_save = []
        batch_size = 100
        
        with tqdm(total=len(chunks), desc="Résumé des chunks", unit="chunk") as pbar:
            for chunk in chunks:
                try:
                    result = summarize_chunk(
                        chunk,
                        self.config.ollama_model,
                        self.config.ollama_url,
                        self.config.max_summary_length
                    )
                except Exception as e:
                    print(f"\n✗ Erreur résumé chunk {chunk.id}: {e}")
                    fallback = chunk.text
                    if len(fallback) > self.config.max_summary_length:
                        fallback = fallback[:self.config.max_summary_length] + "..."
                    result = SummarizedChunk.from_text_chunk(chunk, fallback)
                
                summarized.append(result)
                
                if self.config.save_intermediate:
                    batch_to_save.append(result)
                    if len(batch_to_save) >= batch_size:
                        save_summarized_chunks(batch_to_save, self.config.summaries_dir, append=True, show_progress=False)
                        batch_to_save = []
                
                pbar.update(1)
            
            if self.config.save_intermediate and batch_to_save:
                save_summarized_chunks(batch_to_save, self.config.summaries_dir, append=True, show_progress=False)
        
        return summarized
    
    # -------------------------------------------------------------------------
    # Étape 4: Indexation
    # -------------------------------------------------------------------------
    
    def index(self, summarized_chunks: List[SummarizedChunk]) -> int:
        """Indexe les chunks résumés dans ChromaDB."""
        print("\n=== Indexation dans ChromaDB ===")
        count = self.store.index_summarized_chunks(
            summarized_chunks,
            batch_size=self.config.index_batch_size,
            embedding_batch_size=self.config.embedding_batch_size
        )
        print(f"\n✓ {count} documents indexés")
        return count
    
    # -------------------------------------------------------------------------
    # Exécution de la pipeline
    # -------------------------------------------------------------------------
    
    def run(self) -> int:
        """
        Exécute la pipeline complète selon la source configurée.
        
        Returns:
            Nombre de documents indexés
        """
        # Afficher les stats initiales
        stats_before = self.store.get_collection_stats()
        print(f"Documents existants: {stats_before['total_documents']}")
        
        if self.config.reset_collection:
            print("\n⚠ Réinitialisation de la collection...")
            self.store.reset_collection()
            print("✓ Collection réinitialisée\n")
        elif stats_before['total_documents'] > 0:
            print("  (Les nouveaux documents seront ajoutés aux existants)")
        print()
        
        # Charger/obtenir les données selon la source
        summarized_chunks: List[SummarizedChunk] = []
        
        if self.config.source == PipelineSource.SUMMARIES:
            # Résumés -> Index
            print("=== Chargement depuis les résumés ===")
            summarized_chunks = load_summarized_chunks_from_dir(self.config.summaries_dir)
            if not summarized_chunks:
                print("✗ Aucun résumé trouvé.")
                return 0
            print(f"\n✓ {len(summarized_chunks)} résumés chargés\n")
        
        elif self.config.source == PipelineSource.CHUNKS:
            # Chunks -> Résumés -> Index
            print("=== Chargement depuis les chunks ===")
            chunks = load_text_chunks(self.config.chunks_dir)
            if not chunks:
                print("✗ Aucun chunk trouvé.")
                return 0
            print(f"\n✓ {len(chunks)} chunks chargés\n")
            summarized_chunks = self.summarize(chunks)
            print(f"\n✓ {len(summarized_chunks)} résumés créés")
        
        elif self.config.source == PipelineSource.DOCUMENTS:
            # Documents -> Chunks -> Résumés -> Index
            print("=== Chargement depuis les documents ===")
            documents = load_scraped_documents(self.config.data_dir)
            if not documents:
                print("✗ Aucun document trouvé.")
                return 0
            print(f"\n✓ {len(documents)} documents chargés\n")
            
            chunks = self.chunk_documents(documents)
            summarized_chunks = self.summarize(chunks)
            print(f"\n✓ {len(summarized_chunks)} résumés créés")
        
        else:  # PipelineSource.SCRAPE
            # Scraping -> Documents -> Chunks -> Résumés -> Index
            print("=== Mode: Scraping -> Chunking -> Résumé -> Indexation ===\n")
            documents = asyncio.run(self.scrape())
            if not documents:
                print("✗ Aucun document scrapé.")
                return 0
            print(f"\n✓ {len(documents)} documents scrapés\n")
            
            chunks = self.chunk_documents(documents)
            summarized_chunks = self.summarize(chunks)
            print(f"\n✓ {len(summarized_chunks)} résumés créés")
        
        # Indexation
        count = self.index(summarized_chunks)
        
        # Statistiques finales
        print("\n" + "=" * 60)
        print("=== Statistiques finales ===")
        for key, value in self.store.get_collection_stats().items():
            print(f"  {key}: {value}")
        print()
        print(f"✓ {count} nouveaux documents indexés avec succès!")
        print("=" * 60)
        
        return count


# =============================================================================
# Liste des spiders
# =============================================================================

async def list_spiders_with_counts(rate_limit: int = 5) -> List[tuple]:
    """Liste tous les spiders disponibles avec le nombre de documents."""
    print("=== Récupération de la liste des spiders et du nombre de documents ===\n")
    
    available_spiders = list_available_spiders()
    if not available_spiders:
        print("✗ Aucun spider disponible.")
        return []
    
    config = ScraperConfig(rate_limit=rate_limit)
    spider_info = []
    
    async with EntscheidsucheScraper(config) as scraper:
        print(f"Récupération du nombre de documents pour {len(available_spiders)} spiders...\n")
        
        for i, (spider_id, description) in enumerate(available_spiders.items(), 1):
            try:
                json_paths = await scraper._client.list_documents(spider_id)
                doc_count = len(json_paths) if json_paths else 0
                spider_info.append((spider_id, description, doc_count))
                
                if i % 10 == 0 or i == len(available_spiders):
                    print(f"  Progression: {i}/{len(available_spiders)} spiders traités...")
            except Exception as e:
                spider_info.append((spider_id, description, 0))
                print(f"  ⚠ Erreur pour {spider_id}: {e}")
    
    return spider_info


def display_spiders_list(spider_info: List[tuple]) -> int:
    """Affiche la liste des spiders formatée."""
    if not spider_info:
        print("✗ Aucun spider trouvé.")
        return 1
    
    spider_info.sort(key=lambda x: x[2], reverse=True)
    
    print("\n" + "=" * 80)
    print(f"{'Spider ID':<20} {'Description':<50} {'Documents':>10}")
    print("=" * 80)
    
    total_docs = 0
    for spider_id, description, doc_count in spider_info:
        desc_display = description[:47] + "..." if len(description) > 50 else description
        print(f"{spider_id:<20} {desc_display:<50} {doc_count:>10,}")
        total_docs += doc_count
    
    print("=" * 80)
    print(f"{'TOTAL':<20} {'':<50} {total_docs:>10,}")
    print("=" * 80)
    print()
    
    spiders_with_docs = [s for s in spider_info if s[2] > 0]
    spiders_without_docs = [s for s in spider_info if s[2] == 0]
    
    print("Statistiques:")
    print(f"  - Spiders avec documents: {len(spiders_with_docs)}")
    print(f"  - Spiders sans documents: {len(spiders_without_docs)}")
    print(f"  - Total de documents: {total_docs:,}")
    
    if spiders_with_docs:
        max_spider = max(spiders_with_docs, key=lambda x: x[2])
        print(f"  - Spider avec le plus de documents: {max_spider[0]} ({max_spider[2]:,} documents)")
    
    return 0


# =============================================================================
# CLI
# =============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Crée le parser d'arguments CLI."""
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
        "--list-spiders", action="store_true",
        help="Afficher la liste des spiders disponibles avec le nombre de documents"
    )

    # Source des données
    source_group = parser.add_mutually_exclusive_group(required=False)
    source_group.add_argument(
        "--from-summaries", action="store_true",
        help="Charger depuis data/summaries (résumés déjà créés, pas de scraping)"
    )
    source_group.add_argument(
        "--from-chunks", action="store_true",
        help="Charger depuis data/chunks (chunks déjà créés, pas de scraping)"
    )
    source_group.add_argument(
        "--from-documents", action="store_true",
        help="Charger depuis data/raw (documents déjà scrapés, pas de scraping)"
    )

    # Options de scraping
    parser.add_argument(
        "--spiders", nargs="+", type=str,
        help="Liste des spiders à scraper (ex: CH_BGer VD_FindInfo). Par défaut: tous"
    )
    parser.add_argument(
        "--max-docs", type=int,
        help="Nombre maximum de documents à scraper par spider"
    )
    parser.add_argument(
        "--language", type=str, choices=["de", "fr", "it"],
        help="Filtrer par langue (de, fr, it)"
    )
    parser.add_argument(
        "--only-new", action="store_true",
        help="Ne scraper que les documents qui n'existent pas déjà dans data/raw"
    )
    parser.add_argument(
        "--rate-limit", type=int, default=5,
        help="Limite de requêtes par seconde pour le scraping (défaut: 5)"
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=10,
        help="Nombre maximum de requêtes simultanées (défaut: 10)"
    )

    # Options ChromaDB
    parser.add_argument(
        "--db-path", type=str, default=DEFAULT_DB_PATH,
        help=f"Chemin vers la base ChromaDB (défaut: {DEFAULT_DB_PATH})"
    )
    parser.add_argument(
        "--collection-name", type=str, default=DEFAULT_COLLECTION_NAME,
        help=f"Nom de la collection ChromaDB (défaut: {DEFAULT_COLLECTION_NAME})"
    )
    parser.add_argument(
        "--embedding-model", type=str, default=DEFAULT_EMBEDDING_MODEL,
        help=f"Modèle d'embedding (défaut: {DEFAULT_EMBEDDING_MODEL})"
    )
    parser.add_argument(
        "--device", type=str, default=None, choices=['cpu', 'cuda'],
        help="Device pour le modèle d'embedding (None = auto, 'cpu' ou 'cuda')"
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Réinitialiser la collection avant l'indexation"
    )

    # Options Ollama (pour le résumé)
    parser.add_argument(
        "--ollama-model", type=str, default=DEFAULT_OLLAMA_MODEL,
        help=f"Modèle Ollama pour le résumé (défaut: {DEFAULT_OLLAMA_MODEL})"
    )
    parser.add_argument(
        "--ollama-url", type=str, default="http://localhost:11434",
        help="URL de l'API Ollama (défaut: http://localhost:11434)"
    )
    parser.add_argument(
        "--skip-summarization", action="store_true",
        help="Ne pas résumer. Crée des chunks par tokens avec chevauchement"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=512,
        help="Nombre maximum de tokens par chunk (défaut: 512, limite du modèle E5)"
    )
    parser.add_argument(
        "--max-workers", type=int, default=None,
        help="Nombre maximum de processus parallèles pour le chunking"
    )
    parser.add_argument(
        "--embedding-batch-size", type=int, default=None,
        help="Taille des batches pour les embeddings (défaut: auto)"
    )

    # Options de sauvegarde
    parser.add_argument(
        "--no-save-intermediate", action="store_true",
        help="Ne pas sauvegarder les chunks/résumés intermédiaires"
    )

    # Options de répertoires
    parser.add_argument(
        "--data-dir", type=str, default="data/raw",
        help="Répertoire des documents scrapés (défaut: data/raw)"
    )
    parser.add_argument(
        "--chunks-dir", type=str, default="data/chunks",
        help="Répertoire des chunks (défaut: data/chunks)"
    )
    parser.add_argument(
        "--summaries-dir", type=str, default="data/summaries",
        help="Répertoire des résumés (défaut: data/summaries)"
    )

    return parser


def args_to_config(args: argparse.Namespace) -> PipelineConfig:
    """Convertit les arguments CLI en PipelineConfig."""
    # Déterminer la source
    if args.from_summaries:
        source = PipelineSource.SUMMARIES
    elif args.from_chunks:
        source = PipelineSource.CHUNKS
    elif args.from_documents:
        source = PipelineSource.DOCUMENTS
    else:
        source = PipelineSource.SCRAPE
    
    return PipelineConfig(
        source=source,
        data_dir=args.data_dir,
        chunks_dir=args.chunks_dir,
        summaries_dir=args.summaries_dir,
        spiders=args.spiders,
        max_docs_per_spider=args.max_docs,
        language=args.language,
        only_new=args.only_new,
        rate_limit=args.rate_limit,
        max_concurrent=args.max_concurrent,
        db_path=args.db_path,
        collection_name=args.collection_name,
        embedding_model=args.embedding_model,
        device=args.device,
        reset_collection=args.reset,
        ollama_model=args.ollama_model,
        ollama_url=args.ollama_url,
        skip_summarization=args.skip_summarization,
        max_tokens=args.max_tokens,
        max_workers=args.max_workers,
        embedding_batch_size=args.embedding_batch_size,
        save_intermediate=not args.no_save_intermediate,
    )


def main():
    """Fonction principale avec CLI."""
    parser = create_parser()
    args = parser.parse_args()

    # Si l'option --list-spiders est activée
    if args.list_spiders:
        print("=" * 80)
        print("LawLLM - Liste des spiders disponibles")
        print("=" * 80)
        print()
        spider_info = asyncio.run(list_spiders_with_counts(rate_limit=args.rate_limit))
        return display_spiders_list(spider_info)

    # Afficher le header
    print("=" * 60)
    print("LawLLM - Pipeline d'indexation dans ChromaDB")
    print("=" * 60)
    print()

    # Créer la config et exécuter la pipeline
    config = args_to_config(args)
    pipeline = Pipeline(config)
    count = pipeline.run()
    
    return 0 if count > 0 else 1


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    sys.exit(main())

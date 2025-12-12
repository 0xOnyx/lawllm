"""
LawLLM - RAG pour jugements suisses

Ce package fournit des outils pour scraper, traiter et rechercher dans les
jugements judiciaires suisses depuis entscheidsuche.ch.
"""

# Modèles de données
from .models import (
    DocumentMetadata,
    JobDocument,
    JobsFile,
    ScrapedDocument,
    TextChunk,
    SummarizedChunk,
)

# Scraper
from .scraper import (
    EntscheidsucheScraper,
    ScraperConfig,
    scrape_spider,
    list_available_spiders,
    fetch_spiders_from_api,
    get_spiders_sync,
)

# Processor (chunking, résumé et stockage vectoriel)
from .rag import (
    chunk_document_by_pages,
    chunk_document_by_tokens,
    chunk_documents_batch,
    summarize_chunk,
    summarize_chunks,
    check_ollama_connection,
    list_available_models,
    load_summarized_chunks,
    VectorStore,
    create_vector_store,
)

__version__ = "0.1.0"

__all__ = [
    # Modèles
    "DocumentMetadata",
    "JobDocument",
    "JobsFile",
    "ScrapedDocument",
    "TextChunk",
    "SummarizedChunk",
    # Scraper
    "EntscheidsucheScraper",
    "ScraperConfig",
    "scrape_spider",
    "list_available_spiders",
    "fetch_spiders_from_api",
    "get_spiders_sync",
    # Processor (chunking, résumé et stockage vectoriel)
    "chunk_document_by_pages",
    "chunk_document_by_tokens",
    "chunk_documents_batch",
    "summarize_chunk",
    "summarize_chunks",
    "check_ollama_connection",
    "list_available_models",
    "load_summarized_chunks",
    "VectorStore",
    "create_vector_store",
]

"""
Module de traitement (chunking), résumé et stockage vectoriel des documents.

Ce module contient :
- chunk_document_by_pages: Découpage des documents en chunks d'environ 3 pages
- chunk_document_by_tokens: Découpage des documents en chunks basés sur les tokens avec overlap
- chunk_documents_batch: Découpage parallèle de plusieurs documents avec barre de progression
- summarize_chunk: Résumé d'un chunk via LLM local (Ollama)
- summarize_chunks: Résumé de plusieurs chunks
- check_ollama_connection: Vérification de la connexion Ollama
- list_available_models: Liste des modèles Ollama disponibles
- VectorStore: Gestionnaire de stockage vectoriel avec ChromaDB
"""
from .processor import chunk_document_by_pages, chunk_document_by_tokens, chunk_documents_batch
from .summarizer import (
    summarize_chunk,
    summarize_chunks,
    check_ollama_connection,
    list_available_models,
    load_summarized_chunks,
)
from .vector_store import VectorStore, create_vector_store

__all__ = [
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


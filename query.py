#!/usr/bin/env python3
"""
LawLLM Query - Programme de requête RAG avec reranking.

Ce programme permet de :
1. Charger la base de données vectorielle ChromaDB existante
2. Effectuer une recherche sémantique sur une question
3. Appliquer un reranking avec un cross-encoder
4. Générer une réponse via LLM (Ollama local ou OpenAI)

Usage:
    python query.py "Votre question juridique ici"
    python query.py --interactive  # Mode interactif
    python query.py --model gemma3:12b "Question"  # Avec Ollama
    python query.py --backend openai --model gpt-4o "Question"  # Avec OpenAI
"""
import sys
import argparse
from pathlib import Path

# Ajouter le répertoire racine au PYTHONPATH
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.rag.vector_store import VectorStore, DEFAULT_DB_PATH, DEFAULT_COLLECTION_NAME, DEFAULT_EMBEDDING_MODEL
from src.rag.reranker import Reranker, DEFAULT_RERANKER_MODEL
from src.rag.llm import (
    create_generator,
    OllamaGenerator,
    OpenAIGenerator,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_OLLAMA_URL
)


class RAGQueryEngine:
    """
    Moteur de requête RAG complet avec reranking.
    
    Pipeline:
    1. Recherche sémantique dans ChromaDB (top-k initial)
    2. Reranking avec cross-encoder (top-k final)
    3. Génération de réponse avec LLM (Ollama ou OpenAI)
    """
    
    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        reranker_model: str = DEFAULT_RERANKER_MODEL,
        llm_backend: str = "ollama",
        llm_model: str = None,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        openai_api_key: str = None,
        device: str = None,
        verbose: bool = True
    ):
        """
        Initialise le moteur de requête RAG.
        
        Args:
            db_path: Chemin vers la base ChromaDB
            collection_name: Nom de la collection ChromaDB
            embedding_model: Modèle d'embedding pour la recherche
            reranker_model: Modèle cross-encoder pour le reranking
            llm_backend: Backend LLM ("ollama" ou "openai")
            llm_model: Modèle LLM (None = défaut selon backend)
            ollama_url: URL de l'API Ollama
            openai_api_key: Clé API OpenAI (ou via OPENAI_API_KEY env var)
            device: Device pour les modèles (None = auto, 'cpu', 'cuda')
            verbose: Afficher les messages de progression
        """
        self.verbose = verbose
        
        if verbose:
            print("=" * 60)
            print("  LawLLM Query Engine - Initialisation")
            print("=" * 60)
        
        # Charger la base vectorielle
        if verbose:
            print(f"\n[1/3] Chargement de la base vectorielle...")
        self.vector_store = VectorStore(
            collection_name=collection_name,
            db_path=db_path,
            embedding_model=embedding_model,
            device=device
        )
        
        # Afficher les stats de la collection
        stats = self.vector_store.get_collection_stats()
        if verbose:
            print(f"      → {stats['total_documents']} documents indexés")
        
        # Charger le reranker
        if verbose:
            print(f"\n[2/3] Chargement du modèle de reranking...")
        self.reranker = Reranker(model_name=reranker_model, device=device)
        
        # Charger le générateur LLM
        if llm_model is None:
            llm_model = DEFAULT_OLLAMA_MODEL if llm_backend == "ollama" else DEFAULT_OPENAI_MODEL
        
        if verbose:
            print(f"\n[3/3] Configuration du LLM ({llm_backend}: {llm_model})...")
        
        if llm_backend == "ollama":
            self.generator = OllamaGenerator(
                model=llm_model,
                ollama_url=ollama_url,
                temperature=0.7,
                max_tokens=1024
            )
        elif llm_backend == "openai":
            self.generator = OpenAIGenerator(
                model=llm_model,
                api_key=openai_api_key,
                temperature=0.7,
                max_tokens=1024
            )
        else:
            raise ValueError(f"Unknown LLM backend: {llm_backend}. Use 'ollama' or 'openai'.")
        
        if verbose:
            print("\n" + "=" * 60)
            print("  ✓ Moteur RAG prêt!")
            print("=" * 60 + "\n")
    
    def query(
        self,
        question: str,
        n_initial: int = 20,
        n_rerank: int = 5,
        use_original_text: bool = True,
        include_sources: bool = True,
        stream: bool = False
    ) -> dict:
        """
        Exécute une requête RAG complète.
        
        Args:
            question: La question de l'utilisateur
            n_initial: Nombre de résultats initiaux (avant reranking)
            n_rerank: Nombre de résultats après reranking
            use_original_text: Utiliser le texte original pour le reranking
            include_sources: Inclure les sources dans la réponse
            stream: Retourner la réponse en streaming
        
        Returns:
            Dictionnaire avec la réponse et les métadonnées
        """
        if self.verbose:
            print(f"\n{'─' * 60}")
            print(f"  Question: {question[:80]}{'...' if len(question) > 80 else ''}")
            print(f"{'─' * 60}")
        
        # Étape 1: Recherche sémantique
        if self.verbose:
            print(f"\n[Étape 1] Recherche sémantique (top-{n_initial})...")
        
        search_results = self.vector_store.search(question, n_results=n_initial)
        
        if not search_results:
            return {
                "question": question,
                "answer": "Aucun document pertinent trouvé dans la base de données.",
                "sources": [],
                "n_results": 0
            }
        
        if self.verbose:
            print(f"          → {len(search_results)} documents trouvés")
        
        # Étape 2: Reranking
        if self.verbose:
            print(f"\n[Étape 2] Reranking avec cross-encoder (top-{n_rerank})...")
        
        reranked_results = self.reranker.rerank(
            query=question,
            results=search_results,
            top_k=n_rerank,
            use_original_text=use_original_text
        )
        
        if self.verbose:
            print(f"          → {len(reranked_results)} documents retenus")
            for i, r in enumerate(reranked_results, 1):
                score = r.get('rerank_score', 0)
                doc_id = r.get('metadata', {}).get('document_id', 'N/A')[:30]
                print(f"            {i}. {doc_id}... (score: {score:.3f})")
        
        # Étape 3: Génération
        if self.verbose:
            print(f"\n[Étape 3] Génération de la réponse...")
        
        if stream:
            # En mode streaming, retourner un générateur
            return {
                "question": question,
                "answer_generator": self.generator.generate_stream(
                    question=question,
                    context_documents=reranked_results
                ),
                "sources": reranked_results,
                "n_results": len(reranked_results)
            }
        else:
            answer = self.generator.generate(
                question=question,
                context_documents=reranked_results,
                include_sources=include_sources
            )
            
            return {
                "question": question,
                "answer": answer,
                "sources": reranked_results,
                "n_results": len(reranked_results)
            }
    
    def search_only(
        self,
        question: str,
        n_results: int = 10,
        rerank: bool = True
    ) -> list:
        """
        Effectue uniquement la recherche (sans génération LLM).
        
        Args:
            question: La question de recherche
            n_results: Nombre de résultats à retourner
            rerank: Appliquer le reranking
        
        Returns:
            Liste des documents trouvés
        """
        # Recherche initiale (plus de résultats si reranking)
        n_initial = n_results * 4 if rerank else n_results
        results = self.vector_store.search(question, n_results=n_initial)
        
        if rerank and results:
            results = self.reranker.rerank(
                query=question,
                results=results,
                top_k=n_results
            )
        
        return results


def format_source_links(sources: list) -> str:
    """
    Formate les liens directs (PDF/HTML) des sources.
    
    Args:
        sources: Liste des documents sources avec métadonnées
        
    Returns:
        Chaîne formatée avec les liens
    """
    if not sources:
        return ""
    
    links_parts = []
    for i, src in enumerate(sources, 1):
        meta = src.get('metadata', {})
        url = meta.get('entscheidsuche_url')
        
        if not url:
            continue
        
        # Détecter le type de fichier
        file_type = "Document"
        if url.endswith('.pdf'):
            file_type = "PDF"
        elif url.endswith('.html') or url.endswith('.htm'):
            file_type = "HTML"
        elif url.endswith('.json'):
            file_type = "JSON"
        
        case_number = meta.get('case_number', 'N/A')
        date = meta.get('date', 'N/A')
        
        links_parts.append(f"  [{i}] {case_number} - {date}")
        links_parts.append(f"      📄 {file_type}: {url}")
    
    if links_parts:
        return "\n" + "\n".join(links_parts)
    return ""


def run_interactive(engine: RAGQueryEngine):
    """Mode interactif pour poser des questions."""
    print("\n" + "=" * 60)
    print("  Mode Interactif - LawLLM Query")
    print("  Tapez 'quit' ou 'exit' pour quitter")
    print("  Tapez 'help' pour l'aide")
    print("=" * 60 + "\n")
    
    while True:
        try:
            question = input("\n🔍 Votre question: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ('quit', 'exit', 'q'):
                print("\nAu revoir!")
                break
            
            if question.lower() == 'help':
                print("""
Commandes disponibles:
  - Posez n'importe quelle question juridique
  - 'search: <query>' - Recherche uniquement (sans génération)
  - 'stats' - Afficher les statistiques de la base
  - 'quit' ou 'exit' - Quitter
                """)
                continue
            
            if question.lower() == 'stats':
                stats = engine.vector_store.get_collection_stats()
                print(f"\n📊 Statistiques de la base:")
                print(f"   Collection: {stats['collection_name']}")
                print(f"   Documents: {stats['total_documents']}")
                print(f"   Modèle d'embedding: {stats['embedding_model']}")
                print(f"   Dimension: {stats['embedding_dimension']}")
                continue
            
            if question.lower().startswith('search:'):
                search_query = question[7:].strip()
                results = engine.search_only(search_query, n_results=5)
                print(f"\n📄 {len(results)} résultats trouvés:")
                for i, r in enumerate(results, 1):
                    meta = r.get('metadata', {})
                    print(f"\n  [{i}] {meta.get('case_number', 'N/A')} ({meta.get('date', 'N/A')})")
                    print(f"      Score: {r.get('rerank_score', r.get('distance', 'N/A'))}")
                    summary = r.get('document', '')[:200]
                    print(f"      {summary}...")
                continue
            
            # Requête normale
            result = engine.query(question)
            
            print("\n" + "=" * 60)
            print("  📝 RÉPONSE")
            print("=" * 60)
            print(f"\n{result['answer']}")
            
            # Afficher les sources avec liens directs
            if result['sources']:
                print("\n" + "-" * 60)
                print("  📚 SOURCES")
                print("-" * 60)
                links_text = format_source_links(result['sources'])
                if links_text:
                    print(links_text)
                else:
                    # Fallback si pas de liens formatés
                    for i, src in enumerate(result['sources'], 1):
                        meta = src.get('metadata', {})
                        print(f"  [{i}] {meta.get('case_number', 'N/A')} - {meta.get('date', 'N/A')}")
                        if meta.get('entscheidsuche_url'):
                            print(f"      URL: {meta['entscheidsuche_url']}")
            
        except KeyboardInterrupt:
            print("\n\nInterrompu par l'utilisateur.")
            break
        except Exception as e:
            print(f"\n❌ Erreur: {e}")


def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="LawLLM Query - Recherche sémantique avec reranking et génération LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using Ollama (local)
  python query.py "Quelle est la peine pour le vol en Suisse?"
  python query.py --interactive
  python query.py --model gemma3:12b "Ma question"
  
  # Using OpenAI
  python query.py --backend openai "Ma question"
  python query.py --backend openai --model gpt-4o "Question complexe"
  
  # Search only (no LLM generation)
  python query.py --search-only "Recherche"
        """
    )
    
    # Arguments positionnels
    parser.add_argument(
        "question",
        nargs="?",
        default=None,
        help="La question à poser (optionnel si --interactive)"
    )
    
    # Mode interactif
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Lancer en mode interactif"
    )
    
    # Configuration de la base
    parser.add_argument(
        "--db-path",
        default=DEFAULT_DB_PATH,
        help=f"Chemin vers la base ChromaDB (défaut: {DEFAULT_DB_PATH})"
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION_NAME,
        help=f"Nom de la collection (défaut: {DEFAULT_COLLECTION_NAME})"
    )
    
    # Configuration LLM backend
    parser.add_argument(
        "--backend", "-b",
        choices=["ollama", "openai"],
        default="ollama",
        help="Backend LLM à utiliser (défaut: ollama)"
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help=f"Modèle LLM (défaut: {DEFAULT_OLLAMA_MODEL} pour ollama, {DEFAULT_OPENAI_MODEL} pour openai)"
    )
    parser.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
        help=f"URL de l'API Ollama (défaut: {DEFAULT_OLLAMA_URL})"
    )
    parser.add_argument(
        "--openai-key",
        default=None,
        help="Clé API OpenAI (ou via OPENAI_API_KEY env var)"
    )
    
    # Configuration des modèles embedding/reranking
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"Modèle d'embedding (défaut: {DEFAULT_EMBEDDING_MODEL})"
    )
    parser.add_argument(
        "--reranker-model",
        default=DEFAULT_RERANKER_MODEL,
        help=f"Modèle de reranking (défaut: {DEFAULT_RERANKER_MODEL})"
    )
    
    # Options de recherche
    parser.add_argument(
        "--n-initial",
        type=int,
        default=20,
        help="Nombre de résultats initiaux avant reranking (défaut: 20)"
    )
    parser.add_argument(
        "--n-results", "-n",
        type=int,
        default=5,
        help="Nombre de résultats après reranking (défaut: 5)"
    )
    
    # Options de génération
    parser.add_argument(
        "--no-sources",
        action="store_true",
        help="Ne pas inclure les sources dans la réponse"
    )
    parser.add_argument(
        "--search-only",
        action="store_true",
        help="Recherche uniquement (sans génération LLM)"
    )
    
    # Options système
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="Device pour les modèles (défaut: auto-détection)"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Mode silencieux (moins de messages)"
    )
    
    args = parser.parse_args()
    
    # Vérifier qu'on a une question ou qu'on est en mode interactif
    if not args.interactive and not args.question:
        parser.print_help()
        print("\n❌ Erreur: Spécifiez une question ou utilisez --interactive")
        sys.exit(1)
    
    try:
        # Initialiser le moteur RAG
        engine = RAGQueryEngine(
            db_path=args.db_path,
            collection_name=args.collection,
            embedding_model=args.embedding_model,
            reranker_model=args.reranker_model,
            llm_backend=args.backend,
            llm_model=args.model,
            ollama_url=args.ollama_url,
            openai_api_key=args.openai_key,
            device=args.device,
            verbose=not args.quiet
        )
        
        if args.interactive:
            run_interactive(engine)
        else:
            # Mode question unique
            if args.search_only:
                results = engine.search_only(
                    args.question,
                    n_results=args.n_results
                )
                print(f"\n📄 {len(results)} résultats trouvés:\n")
                for i, r in enumerate(results, 1):
                    meta = r.get('metadata', {})
                    print(f"[{i}] {meta.get('case_number', 'N/A')} ({meta.get('date', 'N/A')})")
                    print(f"    Score: {r.get('rerank_score', r.get('distance', 'N/A'))}")
                    print(f"    {r.get('document', '')[:200]}...\n")
            else:
                result = engine.query(
                    question=args.question,
                    n_initial=args.n_initial,
                    n_rerank=args.n_results,
                    include_sources=not args.no_sources
                )
                
                print("\n" + "=" * 60)
                print("  📝 RÉPONSE")
                print("=" * 60)
                print(f"\n{result['answer']}")
                
                if result['sources'] and not args.no_sources:
                    print("\n" + "-" * 60)
                    print("  📚 SOURCES")
                    print("-" * 60)
                    links_text = format_source_links(result['sources'])
                    if links_text:
                        print(links_text)
                    else:
                        # Fallback si pas de liens formatés
                        for i, src in enumerate(result['sources'], 1):
                            meta = src.get('metadata', {})
                            print(f"  [{i}] {meta.get('case_number', 'N/A')} - {meta.get('date', 'N/A')}")
                            if meta.get('entscheidsuche_url'):
                                print(f"      URL: {meta['entscheidsuche_url']}")
    
    except KeyboardInterrupt:
        print("\n\nInterrompu par l'utilisateur.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

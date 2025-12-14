"""
Module de reranking pour améliorer la pertinence des résultats de recherche.

Ce module utilise un modèle cross-encoder pour re-ordonner les résultats
de la recherche sémantique initiale basée sur la similarité avec la question.
Les chunks provenant du même document sont automatiquement combinés.
"""
import warnings
# Supprimer le warning du tokenizer XLMRoberta (informatif, pas une erreur)
warnings.filterwarnings("ignore", message=".*XLMRobertaTokenizerFast.*")

from typing import List, Dict, Any, Optional
from collections import defaultdict
from FlagEmbedding import FlagReranker
import torch


# Modèle BGE multilingue pour le reranking (meilleure performance multilingue)
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


class Reranker:
    """
    Reranker utilisant un modèle cross-encoder pour améliorer la pertinence.
    
    Le cross-encoder évalue directement la paire (question, document) ce qui
    donne de meilleurs résultats que la recherche par bi-encoder seule.
    
    Les chunks provenant du même document sont automatiquement combinés
    pour fournir un contexte plus complet.
    
    Exemple d'utilisation:
        ```python
        from src.rag.reranker import Reranker
        
        reranker = Reranker()
        results = vector_store.search(query, n_results=20)
        reranked = reranker.rerank(query, results, top_k=5)
        ```
    """
    
    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        device: Optional[str] = None,
        use_fp16: bool = True,
        normalize: bool = True
    ):
        """
        Initialise le reranker avec BGE.
        
        Args:
            model_name: Nom du modèle BGE reranker à utiliser
            device: Device pour le modèle (None = auto-détection GPU, 'cpu', 'cuda')
            use_fp16: Utiliser FP16 pour accélérer le calcul (recommandé avec GPU)
            normalize: Si True, les scores seront normalisés entre 0 et 1 via sigmoid
        """
        self.model_name = model_name
        self.normalize = normalize
        
        # Détecter automatiquement le device si non spécifié
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
                print(f"✓ GPU détecté pour le reranker: {torch.cuda.get_device_name(0)}")
            else:
                device = 'cpu'
                use_fp16 = False  # FP16 n'est pas efficace sur CPU
                print("  Reranker sur CPU")
        
        self.device = device
        self.use_fp16 = use_fp16
        
        # Charger le modèle BGE reranker
        print(f"Chargement du modèle de reranking: {model_name}...")
        self.model = FlagReranker(model_name, use_fp16=use_fp16, device=device)
        print(f"✓ Modèle de reranking chargé")
    
    def _get_text_from_result(self, result: Dict[str, Any], use_original_text: bool) -> str:
        """Extrait le texte d'un résultat."""
        if use_original_text and 'metadata' in result:
            return result['metadata'].get('original_text', result.get('document', ''))
        return result.get('document', '')
    
    def _combine_chunks_by_document(
        self,
        results: List[Dict[str, Any]],
        use_original_text: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Combine les chunks qui proviennent du même document.
        
        Les chunks sont triés par chunk_index et leurs textes sont concaténés.
        Les métadonnées du premier chunk sont conservées.
        
        Args:
            results: Liste des résultats de recherche
            use_original_text: Utiliser le texte original
        
        Returns:
            Liste des documents combinés (un par document_id unique)
        """
        # Regrouper les chunks par document_id
        docs_chunks: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        for result in results:
            doc_id = result.get('metadata', {}).get('document_id', result.get('id', 'unknown'))
            docs_chunks[doc_id].append(result)
        
        combined_results = []
        
        for doc_id, chunks in docs_chunks.items():
            # Trier les chunks par index pour avoir l'ordre correct
            chunks_sorted = sorted(
                chunks,
                key=lambda x: x.get('metadata', {}).get('chunk_index', 0)
            )
            
            # Combiner les textes originaux
            combined_texts = []
            combined_summaries = []
            chunk_indices = []
            
            for chunk in chunks_sorted:
                text = self._get_text_from_result(chunk, use_original_text)
                combined_texts.append(text)
                combined_summaries.append(chunk.get('document', ''))
                chunk_indices.append(chunk.get('metadata', {}).get('chunk_index', 0))
            
            # Créer le document combiné
            # On prend les métadonnées du premier chunk comme base
            first_chunk = chunks_sorted[0]
            combined_metadata = dict(first_chunk.get('metadata', {}))
            
            # Mettre à jour les métadonnées pour refléter la combinaison
            combined_metadata['combined_chunks'] = len(chunks)
            combined_metadata['chunk_indices'] = chunk_indices
            combined_metadata['original_text'] = "\n\n---\n\n".join(combined_texts)
            
            combined_result = {
                'id': doc_id,
                'document': "\n\n".join(combined_summaries),  # Résumés combinés
                'metadata': combined_metadata,
                'distance': min(c.get('distance', float('inf')) for c in chunks),  # Meilleure distance
                'source_chunks': chunks_sorted  # Garder les chunks originaux si besoin
            }
            
            combined_results.append(combined_result)
        
        return combined_results
    
    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        use_original_text: bool = True,
        combine_chunks: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Rerank les résultats de recherche en utilisant le cross-encoder.
        
        Args:
            query: La question/requête de l'utilisateur
            results: Liste des résultats de recherche (format VectorStore.search())
            top_k: Nombre de résultats à retourner après reranking (None = tous)
            use_original_text: Si True, utilise le texte original pour le reranking.
                              Si False, utilise le résumé (document field).
            combine_chunks: Si True, combine les chunks du même document avant le reranking.
        
        Returns:
            Liste des résultats re-ordonnés avec un nouveau score 'rerank_score'
        """
        if not results:
            return []
        
        # Combiner les chunks du même document si demandé
        if combine_chunks:
            n_original = len(results)
            results = self._combine_chunks_by_document(results, use_original_text)
            n_combined = len(results)
            if n_original != n_combined:
                print(f"  [Reranking] {n_original} chunks → {n_combined} documents combinés")
        
        # Préparer les paires [question, texte] pour BGE
        pairs = []
        for result in results:
            text = self._get_text_from_result(result, use_original_text)
            pairs.append([query, text])
        
        # Calculer les scores de reranking avec BGE
        print(f"  [Reranking] Évaluation de {len(pairs)} documents...")
        scores = self.model.compute_score(pairs, normalize=self.normalize)
        
        # compute_score retourne un float si une seule paire, sinon une liste
        if not isinstance(scores, list):
            scores = [scores]
        
        # Ajouter les scores aux résultats
        for i, result in enumerate(results):
            result['rerank_score'] = float(scores[i])
        
        # Trier par score de reranking (décroissant)
        reranked_results = sorted(results, key=lambda x: x['rerank_score'], reverse=True)
        
        # Limiter le nombre de résultats si demandé
        if top_k is not None:
            reranked_results = reranked_results[:top_k]
        
        print(f"  [Reranking] ✓ {len(reranked_results)} documents après reranking")
        return reranked_results


def create_reranker(
    model_name: str = DEFAULT_RERANKER_MODEL,
    device: Optional[str] = None,
    use_fp16: bool = True,
    normalize: bool = True
) -> Reranker:
    """
    Fonction utilitaire pour créer un Reranker.
    
    Args:
        model_name: Nom du modèle BGE reranker à utiliser
        device: Device pour le modèle (None = auto-détection)
        use_fp16: Utiliser FP16 pour accélérer le calcul
        normalize: Si True, scores normalisés entre 0 et 1
        
    Returns:
        Instance de Reranker
    """
    return Reranker(
        model_name=model_name,
        device=device,
        use_fp16=use_fp16,
        normalize=normalize
    )

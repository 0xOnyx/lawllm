"""
Module de stockage vectoriel avec ChromaDB.

Ce module gère :
- La génération d'embeddings avec sentence-transformers (mode local)
- La génération d'embeddings avec Hugging Face Inference API (mode inference)
- L'indexation dans ChromaDB
- La recherche sémantique
"""
import os
from typing import List, Optional, Dict, Any, Literal
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from ..models import SummarizedChunk


# Modèle d'embedding par défaut (multilingue, optimisé pour le français)
DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
DEFAULT_COLLECTION_NAME = "lawllm_chunks"
DEFAULT_DB_PATH = "chroma_db"


class VectorStore:
    """
    Gestionnaire de stockage vectoriel avec ChromaDB.
    
    Supporte deux modes pour la génération d'embeddings:
    - Mode local: utilise sentence-transformers (par défaut)
    - Mode inference: utilise Hugging Face Inference API
    
    Exemple d'utilisation (mode local):
        ```python
        from src.rag.vector_store import VectorStore
        
        store = VectorStore()
        store.index_summarized_chunks(summarized_chunks)
        
        # Recherche
        results = store.search("question juridique", n_results=5)
        ```
    
    Exemple d'utilisation (mode inference API):
        ```python
        from src.rag.vector_store import VectorStore
        
        # Avec token (recommandé)
        store = VectorStore(
            use_inference_api=True,
            inference_api_token="votre_token_hf"
        )
        
        # Ou sans token (rate limité)
        store = VectorStore(use_inference_api=True)
        
        store.index_summarized_chunks(summarized_chunks)
        results = store.search("question juridique", n_results=5)
        ```
    """
    
    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        db_path: str = DEFAULT_DB_PATH,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        device: Optional[str] = None,
        use_inference_api: bool = False,
        inference_api_token: Optional[str] = None,
        inference_api_base_url: Optional[str] = None
    ):
        """
        Initialise le VectorStore.
        
        Args:
            collection_name: Nom de la collection ChromaDB
            db_path: Chemin vers la base de données ChromaDB
            embedding_model: Nom du modèle à utiliser (sentence-transformers ou Hugging Face)
            device: Device pour le modèle local (None = auto-détection GPU, 'cpu', 'cuda')
            use_inference_api: Si True, utilise Hugging Face Inference API au lieu du modèle local
            inference_api_token: Token Hugging Face pour l'API (optionnel, peut être dans env)
            inference_api_base_url: URL de base pour l'API Inference (optionnel)
        """
        self.collection_name = collection_name
        self.db_path = Path(db_path)
        self.embedding_model_name = embedding_model
        self.use_inference_api = use_inference_api
        
        if use_inference_api:
            # Mode Inference API
            from huggingface_hub import InferenceClient
            
            print(f"Mode Inference API activé")
            print(f"Modèle: {embedding_model}")
            
            # Récupérer le token depuis l'argument ou la variable d'environnement
            token = inference_api_token or os.getenv("HUGGINGFACE_API_TOKEN")
            
            # Récupérer la base_url depuis l'argument ou la variable d'environnement
            base_url = inference_api_base_url or os.getenv("HUGGINGFACE_INFERENCE_API_URL")
            
            # Construire les arguments pour InferenceClient
            client_kwargs = {"model": embedding_model}
            if token:
                client_kwargs["token"] = token
            if base_url:
                client_kwargs["base_url"] = base_url
                print(f"  Base URL: {base_url}")
            
            self.inference_client = InferenceClient(**client_kwargs)
            
            if not token:
                print("  ⚠ Aucun token fourni, utilisation de l'API publique (rate limité)")
            
            # Pour l'API, on ne connaît pas la dimension à l'avance, on la détectera à la première génération
            self._embedding_dimension = None
            self.embedding_model = None
            self.device = None
            
            print(f"✓ Client Inference API initialisé")
        else:
            # Mode local avec SentenceTransformer
            # Détecter automatiquement le device si non spécifié
            if device is None:
                try:
                    import torch
                    if torch.cuda.is_available():
                        device = 'cuda'
                        print(f"✓ GPU détecté: {torch.cuda.get_device_name(0)}")
                        print(f"  VRAM disponible: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
                    else:
                        device = 'cpu'
                        print("  Aucun GPU détecté, utilisation du CPU")
                except ImportError:
                    # PyTorch non installé, utiliser CPU
                    device = 'cpu'
                    print("  PyTorch non trouvé, utilisation du CPU")
                except Exception as e:
                    device = 'cpu'
                    print(f"  Erreur lors de la détection GPU: {e}, utilisation du CPU")
            
            # Initialiser le modèle d'embedding
            print(f"Chargement du modèle d'embedding: {embedding_model}...")
            print(f"  Device: {device}")
            self.device = device
            self.embedding_model = SentenceTransformer(embedding_model, device=device)
            
            # Vérifier que le modèle est bien sur le device spécifié
            try:
                import torch
                # Vérifier le device du premier module du modèle
                actual_device = next(self.embedding_model[0].parameters()).device
                print(f"  Device effectif du modèle: {actual_device}")
                if device == 'cuda' and actual_device.type == 'cpu':
                    print("  ⚠ ATTENTION: Le modèle est sur CPU malgré la demande de GPU!")
                    print("    Vérifiez que PyTorch est installé avec support CUDA")
                elif device == 'cuda' and actual_device.type == 'cuda':
                    print(f"  ✓ Modèle chargé sur GPU (device {actual_device})")
            except Exception as e:
                print(f"  ⚠ Impossible de vérifier le device: {e}")
            
            print(f"✓ Modèle chargé (dimension: {self.embedding_model.get_sentence_embedding_dimension()})")
            self._embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()
            self.inference_client = None
        
        # Initialiser ChromaDB
        self._init_chromadb()
    
    def _init_chromadb(self):
        """Initialise la connexion à ChromaDB."""
        # Créer le répertoire si nécessaire
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Initialiser le client ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Obtenir ou créer la collection
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            print(f"✓ Collection '{self.collection_name}' trouvée ({self.collection.count()} documents)")
        except Exception:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "LawLLM - Résumés de jugements suisses"}
            )
            print(f"✓ Collection '{self.collection_name}' créée")
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 128, show_progress: bool = True) -> List[List[float]]:
        """
        Génère les embeddings pour une liste de textes.
        
        Args:
            texts: Liste des textes à encoder
            batch_size: Taille des batches pour le traitement (défaut: 128, augmenté pour de meilleures performances)
            show_progress: Afficher la barre de progression
            
        Returns:
            Liste des embeddings (chaque embedding est une liste de floats)
        """
        if not texts:
            return []
        
        if self.use_inference_api:
            # Mode Inference API - Optimisé pour 300+ embeddings/s
            print(f"\n  [Embeddings] Mode Inference API (optimisé pour haute performance)")
            print(f"  [Embeddings] Nombre de textes à encoder: {len(texts)}")
            print(f"  [Embeddings] Début de la génération des embeddings via API...\n")
            
            # Optimisation pour haute performance : batches plus grands et traitement parallèle
            # Pour atteindre 300+ embeddings/s, on utilise des batches de 100-200 avec parallélisme
            effective_batch_size = max(batch_size, 100)  # Batch minimum de 100 pour optimiser le débit
            max_workers = 10  # Nombre de threads parallèles pour les appels API
            
            embeddings = []
            start_time = time.time()
            
            def process_batch(batch_idx: int, batch_texts: List[str]) -> tuple:
                """Traite un batch d'embeddings."""
                try:
                    # Utiliser la méthode feature_extraction de InferenceClient
                    response = self.inference_client.feature_extraction(batch_texts)
                    
                    # La réponse est une liste de listes (un embedding par texte)
                    import numpy as np
                    if isinstance(response, np.ndarray):
                        batch_embeddings_list = response.tolist()
                    elif isinstance(response, list):
                        # Si c'est une liste de listes
                        if len(response) > 0 and isinstance(response[0], list):
                            batch_embeddings_list = response
                        else:
                            # Si c'est une seule liste (un seul embedding)
                            batch_embeddings_list = [response]
                    else:
                        # Autre format
                        if hasattr(response, 'tolist'):
                            batch_embeddings_list = response.tolist()
                        else:
                            batch_embeddings_list = [list(response)]
                    
                    return batch_idx, batch_embeddings_list, None
                except Exception as e:
                    return batch_idx, None, e
            
            # Créer les batches
            batches = []
            for i in range(0, len(texts), effective_batch_size):
                batch_texts = texts[i:i + effective_batch_size]
                batches.append((i // effective_batch_size, batch_texts))
            
            # Traiter les batches en parallèle pour maximiser le débit
            batch_results = {}
            
            if show_progress:
                pbar = tqdm(total=len(batches), desc="Génération embeddings (API)", unit="batch")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Soumettre toutes les tâches
                future_to_batch = {
                    executor.submit(process_batch, batch_idx, batch_texts): batch_idx
                    for batch_idx, batch_texts in batches
                }
                
                # Traiter les résultats au fur et à mesure
                for future in as_completed(future_to_batch):
                    batch_idx, batch_embeddings, error = future.result()
                    
                    if error:
                        print(f"\n  ⚠ Erreur lors de la génération d'embeddings pour le batch {batch_idx}: {error}")
                        raise error
                    
                    if batch_embeddings:
                        batch_results[batch_idx] = batch_embeddings
                    
                    if show_progress:
                        pbar.update(1)
            
            if show_progress:
                pbar.close()
            
            # Reconstruire les embeddings dans l'ordre
            for batch_idx in sorted(batch_results.keys()):
                embeddings.extend(batch_results[batch_idx])
            
            # Détecter la dimension à la première génération
            if self._embedding_dimension is None and embeddings:
                self._embedding_dimension = len(embeddings[0])
                print(f"  [Embeddings] Dimension détectée: {self._embedding_dimension}")
            
            # Normaliser les embeddings (comme pour le mode local)
            import numpy as np
            embeddings_array = np.array(embeddings)
            norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)  # Éviter division par zéro
            embeddings_normalized = embeddings_array / norms
            
            # Calculer et afficher les performances
            elapsed_time = time.time() - start_time
            embeddings_per_second = len(embeddings) / elapsed_time if elapsed_time > 0 else 0
            
            print(f"\n  [Embeddings] ✓ {len(embeddings)} embeddings générés via Inference API")
            print(f"  [Embeddings] ⚡ Performance: {embeddings_per_second:.1f} embeddings/s ({elapsed_time:.2f}s)")
            
            return embeddings_normalized.tolist()
        
        else:
            # Mode local avec SentenceTransformer
            # Optimisation: utiliser un batch_size plus grand pour de meilleures performances
            # Ajuster automatiquement selon la taille des données
            if len(texts) > 10000:
                # Pour de très gros volumes, augmenter encore le batch_size
                effective_batch_size = max(batch_size, 256)
            else:
                effective_batch_size = batch_size
            
            # Vérifier et afficher le device utilisé
            try:
                import torch
                model_device = next(self.embedding_model[0].parameters()).device
                device_type = model_device.type
                device_str = str(model_device)
                
                print(f"\n  [Embeddings] Device demandé: {self.device}")
                print(f"  [Embeddings] Device effectif du modèle: {device_str}")
                
                if device_type == 'cuda':
                    # Afficher des infos GPU
                    gpu_name = torch.cuda.get_device_name(model_device.index if model_device.index is not None else 0)
                    memory_allocated = torch.cuda.memory_allocated(model_device.index if model_device.index is not None else 0) / 1024**3
                    memory_reserved = torch.cuda.memory_reserved(model_device.index if model_device.index is not None else 0) / 1024**3
                    print(f"  [Embeddings] ✓ GPU actif: {gpu_name}")
                    print(f"  [Embeddings]   VRAM utilisée: {memory_allocated:.2f} GB / {memory_reserved:.2f} GB réservée")
                else:
                    print(f"  [Embeddings] ⚠ CPU utilisé (plus lent)")
                    if self.device == 'cuda':
                        print(f"  [Embeddings]   ATTENTION: GPU demandé mais modèle sur CPU!")
                        if torch.cuda.is_available():
                            print(f"  [Embeddings]   GPU disponible mais non utilisé. Tentative de transfert...")
                            try:
                                self.embedding_model = self.embedding_model.to('cuda')
                                model_device = next(self.embedding_model[0].parameters()).device
                                print(f"  [Embeddings]   ✓ Modèle transféré sur GPU: {model_device}")
                            except Exception as transfer_error:
                                print(f"  [Embeddings]   ✗ Échec du transfert sur GPU: {transfer_error}")
                        else:
                            print(f"  [Embeddings]   Aucun GPU disponible")
                
                print(f"  [Embeddings] Batch size: {effective_batch_size}")
                print(f"  [Embeddings] Nombre de textes à encoder: {len(texts)}")
                
            except ImportError:
                print(f"  [Embeddings] ⚠ PyTorch non disponible, impossible de vérifier le device")
                print(f"  [Embeddings] Device configuré: {self.device}")
            except Exception as e:
                print(f"  [Embeddings] ⚠ Erreur lors de la vérification du device: {e}")
                print(f"  [Embeddings] Device configuré: {self.device}")
            
            # Générer les embeddings par batch (plus efficace)
            print(f"  [Embeddings] Début de la génération des embeddings...\n")
            
            encode_kwargs = {
                "batch_size": effective_batch_size,
                "convert_to_numpy": True,
                "normalize_embeddings": True,  # Normaliser pour de meilleures performances
                "show_progress_bar": show_progress
                # Note: device n'est pas passé car encode() utilise automatiquement le device du modèle
            }
            
            embeddings = self.embedding_model.encode(texts, **encode_kwargs)
            
            # Afficher un message de confirmation après génération
            try:
                import torch
                if torch.cuda.is_available():
                    model_device = next(self.embedding_model[0].parameters()).device
                    if model_device.type == 'cuda':
                        memory_allocated = torch.cuda.memory_allocated(model_device.index if model_device.index is not None else 0) / 1024**3
                        print(f"\n  [Embeddings] ✓ {len(embeddings)} embeddings générés sur GPU")
                        print(f"  [Embeddings]   VRAM utilisée après génération: {memory_allocated:.2f} GB")
                    else:
                        print(f"\n  [Embeddings] ✓ {len(embeddings)} embeddings générés sur CPU")
                else:
                    print(f"\n  [Embeddings] ✓ {len(embeddings)} embeddings générés sur CPU")
            except:
                print(f"\n  [Embeddings] ✓ {len(embeddings)} embeddings générés")
            
            # Optimisation: conversion plus efficace pour de gros volumes
            # ChromaDB accepte les arrays numpy, mais on convertit en list pour compatibilité
            if len(embeddings) > 10000:
                # Pour de très gros volumes, convertir en chunks pour éviter la surcharge mémoire
                print(f"  [Embeddings] Conversion des embeddings (gros volume, traitement par chunks)...")
                result = []
                chunk_size = 1000
                for i in range(0, len(embeddings), chunk_size):
                    result.extend([emb.tolist() for emb in embeddings[i:i + chunk_size]])
                return result
            else:
                return [embedding.tolist() for embedding in embeddings]
    
    def index_summarized_chunks(
        self,
        summarized_chunks: List[SummarizedChunk],
        batch_size: int = 128,
        embedding_batch_size: Optional[int] = None,
        reset_collection: bool = False
    ) -> int:
        """
        Indexe une liste de SummarizedChunk dans ChromaDB.
        
        Args:
            summarized_chunks: Liste des chunks résumés à indexer
            batch_size: Taille des batches pour l'indexation dans ChromaDB (défaut: 128)
            embedding_batch_size: Taille des batches pour la génération d'embeddings (None = auto)
            reset_collection: Si True, réinitialise la collection avant l'indexation
            
        Returns:
            Nombre de documents indexés
        """
        if not summarized_chunks:
            print("Aucun chunk à indexer")
            return 0
        
        # Réinitialiser la collection si demandé
        if reset_collection:
            print(f"⚠ Réinitialisation de la collection '{self.collection_name}'...")
            try:
                self.client.delete_collection(name=self.collection_name)
            except Exception:
                pass
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "LawLLM - Résumés de jugements suisses"}
            )
            print("✓ Collection réinitialisée")
        
        # Extraire les résumés pour les embeddings
        summaries = [chunk.summary for chunk in summarized_chunks]
        
        # Générer les embeddings avec batch_size optimisé
        if embedding_batch_size is None:
            # Ajuster automatiquement selon le nombre de documents
            if len(summaries) > 50000:
                embedding_batch_size = 512  # Très gros volumes
            elif len(summaries) > 10000:
                embedding_batch_size = 256  # Gros volumes
            else:
                embedding_batch_size = 128  # Volumes normaux
        
        print(f"Génération des embeddings pour {len(summaries)} résumés (batch_size: {embedding_batch_size})...")
        embeddings = self.generate_embeddings(summaries, batch_size=embedding_batch_size)
        
        # Préparer les données pour ChromaDB
        ids = [chunk.id for chunk in summarized_chunks]
        documents = summaries  # Les documents sont les résumés
        metadatas = [chunk.to_chromadb_metadata() for chunk in summarized_chunks]
        
        # Indexer dans ChromaDB par batch
        print(f"Indexation dans ChromaDB...")
        total_indexed = 0
        
        for i in tqdm(range(0, len(ids), batch_size), desc="Indexation"):
            batch_ids = ids[i:i + batch_size]
            batch_documents = documents[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]
            
            self.collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_documents,
                metadatas=batch_metadatas
            )
            total_indexed += len(batch_ids)
        
        print(f"✓ {total_indexed} documents indexés dans ChromaDB")
        return total_indexed
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Recherche sémantique dans la collection.
        
        Args:
            query: Requête de recherche
            n_results: Nombre de résultats à retourner
            filter_metadata: Filtres à appliquer sur les métadonnées (ex: {"spider": "CH_BGer"})
            
        Returns:
            Liste des résultats avec leurs métadonnées et scores de similarité
        """
        # Générer l'embedding de la requête
        if self.use_inference_api:
            # Mode Inference API
            try:
                # Utiliser la méthode feature_extraction de InferenceClient
                response = self.inference_client.feature_extraction(query)
                
                # La réponse est un embedding (liste de floats ou numpy array)
                import numpy as np
                if isinstance(response, np.ndarray):
                    query_embedding_raw = response.tolist()
                elif isinstance(response, list):
                    query_embedding_raw = response
                elif hasattr(response, 'tolist'):
                    query_embedding_raw = response.tolist()
                else:
                    query_embedding_raw = list(response)
                
                # Normaliser l'embedding
                query_embedding_array = np.array(query_embedding_raw)
                norm = np.linalg.norm(query_embedding_array)
                if norm > 0:
                    query_embedding_array = query_embedding_array / norm
                query_embedding = query_embedding_array.tolist()
            except Exception as e:
                raise Exception(f"Erreur lors de la génération de l'embedding de requête via API: {e}")
        else:
            # Mode local
            query_embedding = self.embedding_model.encode(
                query,
                convert_to_numpy=True,
                normalize_embeddings=True
            ).tolist()
        
        # Construire le filtre where si nécessaire
        where = filter_metadata if filter_metadata else None
        
        # Rechercher dans ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where
        )
        
        # Formater les résultats
        formatted_results = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                result = {
                    'id': results['ids'][0][i],
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                }
                formatted_results.append(result)
        
        return formatted_results
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Retourne les statistiques de la collection.
        
        Returns:
            Dictionnaire avec les statistiques
        """
        count = self.collection.count()
        
        if self.use_inference_api:
            embedding_dimension = self._embedding_dimension or "non détecté"
        else:
            embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()
        
        return {
            'collection_name': self.collection_name,
            'total_documents': count,
            'embedding_model': self.embedding_model_name,
            'embedding_mode': 'inference_api' if self.use_inference_api else 'local',
            'embedding_dimension': embedding_dimension,
            'db_path': str(self.db_path)
        }
    
    def reset_collection(self):
        """Réinitialise complètement la collection."""
        print(f"⚠ Réinitialisation de la collection '{self.collection_name}'...")
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "LawLLM - Résumés de jugements suisses"}
        )
        print("✓ Collection réinitialisée")


def create_vector_store(
    collection_name: str = DEFAULT_COLLECTION_NAME,
    db_path: str = DEFAULT_DB_PATH,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    use_inference_api: bool = False,
    inference_api_token: Optional[str] = None,
    inference_api_base_url: Optional[str] = None
) -> VectorStore:
    """
    Fonction utilitaire pour créer un VectorStore.
    
    Args:
        collection_name: Nom de la collection ChromaDB
        db_path: Chemin vers la base de données ChromaDB
        embedding_model: Nom du modèle à utiliser (sentence-transformers ou Hugging Face)
        use_inference_api: Si True, utilise Hugging Face Inference API au lieu du modèle local
        inference_api_token: Token Hugging Face pour l'API (optionnel, peut être dans env)
        inference_api_base_url: URL de base pour l'API Inference (optionnel)
        
    Returns:
        Instance de VectorStore
    """
    return VectorStore(
        collection_name=collection_name,
        db_path=db_path,
        embedding_model=embedding_model,
        use_inference_api=use_inference_api,
        inference_api_token=inference_api_token,
        inference_api_base_url=inference_api_base_url
    )


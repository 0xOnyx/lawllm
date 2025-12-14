"""
Module de découpage des documents par pages et par tokens.

Ce module permet de découper les documents en chunks d'environ 3 pages
ou en chunks basés sur le nombre de tokens avec chevauchement.

Note: Ce module utilise le tokenizer du modèle E5 (intfloat/multilingual-e5-base)
pour garantir que les chunks respectent la limite de 512 tokens du modèle.
"""
from typing import List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from transformers import AutoTokenizer
from ..models import ScrapedDocument, TextChunk


# Estimation: 1 page A4 ≈ 2500-3000 caractères
# Pour 3 pages: environ 8000 caractères
CHARS_PER_PAGE = 2700
PAGES_PER_CHUNK = 3
CHARS_PER_CHUNK = CHARS_PER_PAGE * PAGES_PER_CHUNK

# Modèle E5 par défaut (doit correspondre à celui utilisé dans vector_store.py)
DEFAULT_E5_MODEL = "intfloat/multilingual-e5-base"
# Limite de tokens du modèle E5
E5_MAX_TOKENS = 512


def chunk_document_by_pages(
    document: ScrapedDocument,
    pages_per_chunk: int = PAGES_PER_CHUNK,
    chars_per_page: int = CHARS_PER_PAGE
) -> List[TextChunk]:
    """
    Découpe un document en chunks d'environ N pages.
    
    Args:
        document: Document à découper
        pages_per_chunk: Nombre de pages par chunk (défaut: 3)
        chars_per_page: Nombre de caractères estimés par page (défaut: 2700)
        
    Returns:
        Liste de TextChunk avec métadonnées
    """
    if not document.content or len(document.content.strip()) == 0:
        return []
    
    chunks = []
    text = document.content
    chunk_size = pages_per_chunk * chars_per_page
    chunk_index = 0
    
    # Découper le texte en chunks de taille approximative
    start = 0
    text_length = len(text)
    
    while start < text_length:
        # Calculer la fin du chunk
        end = min(start + chunk_size, text_length)
        
        # Si on n'est pas à la fin du texte, essayer de couper à un espace ou saut de ligne
        if end < text_length:
            # Chercher le dernier saut de ligne dans les 500 derniers caractères
            search_start = max(start, end - 500)
            last_newline = text.rfind('\n', search_start, end)
            
            if last_newline > start:
                end = last_newline + 1
            else:
                # Sinon, chercher le dernier espace
                last_space = text.rfind(' ', search_start, end)
                if last_space > start:
                    end = last_space + 1
        
        # Extraire le chunk
        chunk_text = text[start:end].strip()
        
        # Ne créer un chunk que s'il n'est pas vide
        if chunk_text:
            chunk = _create_chunk(chunk_text, document, chunk_index)
            chunks.append(chunk)
            chunk_index += 1
        
        # Passer au chunk suivant
        start = end
    
    return chunks


# Cache global pour le tokenizer (partagé entre processus)
_tokenizer_cache = {}


def _get_tokenizer(model_name: str = DEFAULT_E5_MODEL):
    """
    Récupère ou crée un tokenizer HuggingFace (mise en cache).
    
    Args:
        model_name: Nom du modèle HuggingFace (défaut: multilingual-e5-base)
        
    Returns:
        AutoTokenizer pour le modèle spécifié
    """
    if model_name not in _tokenizer_cache:
        # Charger le tokenizer sans limite de longueur (on gère nous-mêmes le découpage)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Désactiver l'avertissement de longueur max (on découpe manuellement après)
        tokenizer.model_max_length = 1_000_000  # Valeur très grande pour éviter les warnings
        _tokenizer_cache[model_name] = tokenizer
    return _tokenizer_cache[model_name]


def chunk_document_by_tokens(
    document: ScrapedDocument,
    max_tokens: int = E5_MAX_TOKENS,
    overlap_ratio: float = 0.2,
    model_name: str = DEFAULT_E5_MODEL,
    tokenizer = None
) -> List[TextChunk]:
    """
    Découpe un document en chunks basés sur le nombre de tokens avec chevauchement.
    
    Cette fonction utilise le tokenizer du modèle E5 pour compter les tokens et crée 
    des chunks avec un overlap calculé automatiquement selon le ratio spécifié.
    
    Args:
        document: Document à découper
        max_tokens: Nombre maximum de tokens par chunk (défaut: 512 pour E5)
        overlap_ratio: Ratio d'overlap entre chunks (défaut: 0.2 = 20%)
                      Exemple: max_tokens=512, overlap_ratio=0.2 → overlap=102 tokens
        model_name: Nom du modèle HuggingFace pour le tokenizer (défaut: multilingual-e5-base)
        tokenizer: Tokenizer pré-initialisé (optionnel, pour éviter de le recréer)
        
    Returns:
        Liste de TextChunk avec métadonnées
    """
    if not document.content or len(document.content.strip()) == 0:
        return []
    
    # Utiliser le tokenizer fourni ou en créer un nouveau
    if tokenizer is None:
        tokenizer = _get_tokenizer(model_name)
    
    # Calculer l'overlap en tokens
    overlap_tokens = int(max_tokens * overlap_ratio)
    
    # Encoder le texte complet en tokens
    text = document.content
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    if len(tokens) == 0:
        return []
    
    chunks = []
    chunk_index = 0
    start = 0
    
    while start < len(tokens):
        # Calculer la fin du chunk
        end = min(start + max_tokens, len(tokens))
        
        # Extraire les tokens du chunk
        chunk_tokens = tokens[start:end]
        
        # Décoder les tokens en texte
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        
        # Créer le chunk
        chunk = _create_chunk(chunk_text, document, chunk_index)
        chunks.append(chunk)
        chunk_index += 1
        
        # Avancer avec chevauchement
        if end >= len(tokens):
            break
        
        # Calculer la position de départ du prochain chunk avec overlap
        new_start = end - overlap_tokens
        
        # S'assurer qu'on avance toujours (éviter les boucles infinies)
        if new_start <= start:
            new_start = start + 1
        
        start = new_start
    
    return chunks


def _chunk_document_worker(args):
    """
    Fonction worker pour le traitement parallèle.
    Prend un tuple (document_dict, max_tokens, overlap_ratio, model_name).
    """
    document_dict, max_tokens, overlap_ratio, model_name = args
    
    # Reconstruire le document depuis le dictionnaire
    document = ScrapedDocument(**document_dict)
    
    # Chunker le document
    return chunk_document_by_tokens(
        document,
        max_tokens=max_tokens,
        overlap_ratio=overlap_ratio,
        model_name=model_name
    )


def chunk_documents_batch(
    documents: List[ScrapedDocument],
    max_tokens: int = E5_MAX_TOKENS,
    overlap_ratio: float = 0.2,
    model_name: str = DEFAULT_E5_MODEL,
    max_workers: Optional[int] = None,
    desc: str = "Chunking documents"
) -> List[TextChunk]:
    """
    Découpe plusieurs documents en chunks en parallèle avec barre de progression.
    
    Cette fonction utilise ProcessPoolExecutor pour paralléliser le chunking
    et affiche une barre de progression avec tqdm.
    
    Args:
        documents: Liste de documents à découper
        max_tokens: Nombre maximum de tokens par chunk (défaut: 512 pour E5)
        overlap_ratio: Ratio d'overlap entre chunks (défaut: 0.2 = 20%)
        model_name: Nom du modèle HuggingFace pour le tokenizer (défaut: multilingual-e5-base)
        max_workers: Nombre maximum de processus parallèles (None = nombre de CPUs)
        desc: Description pour la barre de progression
        
    Returns:
        Liste de tous les TextChunk de tous les documents
    """
    if not documents:
        return []
    
    print(f"  [Chunking] Utilisation du tokenizer: {model_name}")
    print(f"  [Chunking] Max tokens par chunk: {max_tokens}")
    print(f"  [Chunking] Overlap ratio: {overlap_ratio} ({int(max_tokens * overlap_ratio)} tokens)")
    
    # Convertir les documents en dictionnaires pour la sérialisation
    document_dicts = [doc.model_dump() for doc in documents]
    
    # Préparer les arguments pour chaque worker
    args_list = [
        (doc_dict, max_tokens, overlap_ratio, model_name)
        for doc_dict in document_dicts
    ]
    
    all_chunks = []
    
    # Utiliser ProcessPoolExecutor pour paralléliser le traitement
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Soumettre toutes les tâches
        future_to_doc = {
            executor.submit(_chunk_document_worker, args): i
            for i, args in enumerate(args_list)
        }
        
        # Traiter les résultats au fur et à mesure avec barre de progression
        with tqdm(total=len(documents), desc=desc, unit="doc") as pbar:
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
    
    return all_chunks


def _create_chunk(
    text: str,
    document: ScrapedDocument,
    chunk_index: int
) -> TextChunk:
    """
    Crée un chunk avec les métadonnées du document.
    
    Args:
        text: Texte du chunk
        document: Document source
        chunk_index: Index du chunk dans le document
        
    Returns:
        TextChunk avec toutes les métadonnées
    """
    chunk_id = f"{document.id}_chunk_{chunk_index}"
    
    return TextChunk(
        id=chunk_id,
        text=text,
        document_id=document.id,
        chunk_index=chunk_index,
        spider=document.spider,
        language=document.language,
        date=document.date,
        case_number=document.case_number,
        title=document.title,
        entscheidsuche_url=document.entscheidsuche_url
    )


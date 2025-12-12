"""
Module de résumé des chunks via un LLM local.

Ce module permet de résumer chaque chunk de texte en utilisant un LLM local
(via Ollama) pour créer des résumés courts et structurés.
"""
import requests
from typing import Optional
from ..models import TextChunk, SummarizedChunk


# Configuration par défaut
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "gemma3:4b-it-qat"  # Modèle par défaut, peut être changé
MAX_SUMMARY_LENGTH = 500  # Maximum 500 caractères


def summarize_chunk(
    chunk: TextChunk,
    model: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    max_length: int = MAX_SUMMARY_LENGTH
) -> SummarizedChunk:
    """
    Résume un chunk de texte en utilisant un LLM local (Ollama).
    
    Args:
        chunk: Le chunk de texte à résumer
        model: Nom du modèle Ollama à utiliser (défaut: "gemma3:4b-it-qat")
        ollama_url: URL de l'API Ollama (défaut: "http://localhost:11434")
        max_length: Longueur maximale du résumé en caractères (défaut: 500)
        
    Returns:
        SummarizedChunk contenant le résumé et toutes les métadonnées
        
    Raises:
        requests.exceptions.RequestException: Si la connexion à Ollama échoue
        ValueError: Si le résumé généré est vide
    """
    prompt = f"""You are a legal expert. Summarize the following text in a legal and structured manner. 
Keep the summary concise and focus on key legal points, facts, and conclusions.
Maximum length: {max_length} characters.

IMPORTANT: Provide ONLY the summary text. Do not include any prefix, title, label, or formatting like "Summary:", "**Summary:**", or similar. Start directly with the summary content.

Text to summarize:
{chunk.text}

Summary:"""
    
    # Préparer la requête pour Ollama
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,  # Température basse pour des résumés plus factuels
            "max_tokens": 200,   # Limiter les tokens pour garder le résumé court
        }
    }
    
    try:
        # Envoyer la requête à Ollama
        response = requests.post(
            f"{ollama_url}/api/generate",
            json=payload,
            timeout=120  # Timeout de 2 minutes
        )
        response.raise_for_status()
        
        # Extraire le résumé de la réponse
        result = response.json()
        summary = result.get("response", "").strip()
        
        if not summary:
            raise ValueError("Le LLM a retourné un résumé vide")
        
        # Nettoyer les préfixes courants que le modèle pourrait ajouter
        prefixes_to_remove = [
            "**Summary:**",
            "**Summary**",
            "Summary:",
            "Summary",
            "Résumé:",
            "Résumé",
            "**Résumé:**",
            "**Résumé**",
        ]
        
        for prefix in prefixes_to_remove:
            if summary.startswith(prefix):
                summary = summary[len(prefix):].strip()
                # Supprimer aussi les deux-points ou tirets qui pourraient rester
                if summary.startswith(":") or summary.startswith("-"):
                    summary = summary[1:].strip()
                break
        
        # Tronquer le résumé si nécessaire pour respecter la limite
        if len(summary) > max_length:
            # Tronquer à la dernière phrase complète si possible
            truncated = summary[:max_length]
            last_period = truncated.rfind('.')
            last_exclamation = truncated.rfind('!')
            last_question = truncated.rfind('?')
            
            last_sentence_end = max(last_period, last_exclamation, last_question)
            if last_sentence_end > max_length * 0.7:  # Si on trouve une fin de phrase dans les 70% derniers
                summary = truncated[:last_sentence_end + 1]
            else:
                summary = truncated + "..."
        
        # Créer et retourner le SummarizedChunk avec toutes les métadonnées
        return SummarizedChunk.from_text_chunk(chunk, summary)
        
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            f"Impossible de se connecter à Ollama à {ollama_url}. "
            "Assurez-vous qu'Ollama est démarré et accessible."
        )
    except requests.exceptions.Timeout:
        raise TimeoutError(
            f"Timeout lors de la requête à Ollama. Le modèle {model} peut être trop lent."
        )
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            raise ValueError(
                f"Modèle '{model}' non trouvé dans Ollama. "
                "Utilisez 'ollama pull {model}' pour télécharger le modèle."
            )
        raise


def summarize_chunks(
    chunks: list[TextChunk],
    model: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    max_length: int = MAX_SUMMARY_LENGTH
) -> list[SummarizedChunk]:
    """
    Résume plusieurs chunks en utilisant un LLM local.
    
    Args:
        chunks: Liste des chunks à résumer
        model: Nom du modèle Ollama à utiliser
        ollama_url: URL de l'API Ollama
        max_length: Longueur maximale de chaque résumé en caractères
        
    Returns:
        Liste des SummarizedChunk dans le même ordre que les chunks
    """
    summarized_chunks = []
    
    for i, chunk in enumerate(chunks):
        try:
            summarized_chunk = summarize_chunk(chunk, model, ollama_url, max_length)
            summarized_chunks.append(summarized_chunk)
            print(f"  ✓ Chunk {i+1}/{len(chunks)} résumé ({len(summarized_chunk.summary)} caractères)")
        except Exception as e:
            print(f"  ✗ Erreur lors du résumé du chunk {i+1}: {e}")
            # En cas d'erreur, utiliser un résumé tronqué du texte original
            fallback_summary = chunk.text[:max_length]
            if len(chunk.text) > max_length:
                fallback_summary += "..."
            # Créer un SummarizedChunk avec le fallback
            fallback_chunk = SummarizedChunk.from_text_chunk(chunk, fallback_summary)
            summarized_chunks.append(fallback_chunk)
    
    return summarized_chunks


def check_ollama_connection(ollama_url: str = DEFAULT_OLLAMA_URL) -> bool:
    """
    Vérifie si Ollama est accessible et fonctionne.
    
    Args:
        ollama_url: URL de l'API Ollama
        
    Returns:
        True si Ollama est accessible, False sinon
    """
    try:
        response = requests.get(f"{ollama_url}/api/tags", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def list_available_models(ollama_url: str = DEFAULT_OLLAMA_URL) -> list[str]:
    """
    Liste les modèles disponibles dans Ollama.
    
    Args:
        ollama_url: URL de l'API Ollama
        
    Returns:
        Liste des noms de modèles disponibles
        
    Raises:
        requests.exceptions.RequestException: Si la connexion à Ollama échoue
    """
    try:
        response = requests.get(f"{ollama_url}/api/tags", timeout=10)
        response.raise_for_status()
        
        data = response.json()
        models = [model["name"] for model in data.get("models", [])]
        return models
    except requests.exceptions.RequestException as e:
        raise ConnectionError(
            f"Impossible de récupérer la liste des modèles depuis Ollama: {e}"
        )


def load_summarized_chunks(filepath: str) -> list[SummarizedChunk]:
    """
    Charge des SummarizedChunk depuis un fichier JSON.
    
    Args:
        filepath: Chemin vers le fichier JSON contenant les SummarizedChunk
        
    Returns:
        Liste de SummarizedChunk chargés depuis le fichier
    """
    import json
    from pathlib import Path
    
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Le fichier {filepath} n'existe pas")
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Si c'est une liste, charger directement
    if isinstance(data, list):
        return [SummarizedChunk(**item) for item in data]
    # Sinon, essayer de trouver une clé contenant les chunks
    elif isinstance(data, dict):
        # Chercher une clé qui contient une liste
        for key, value in data.items():
            if isinstance(value, list):
                return [SummarizedChunk(**item) for item in value]
        raise ValueError(f"Format de fichier non reconnu dans {filepath}")
    else:
        raise ValueError(f"Format de fichier non reconnu dans {filepath}")


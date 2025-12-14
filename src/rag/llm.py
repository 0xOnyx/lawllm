"""
Module de génération de réponses avec LLM (Ollama local ou OpenAI).

Ce module permet de générer des réponses aux questions juridiques
en utilisant le contexte récupéré par la recherche sémantique.
Supporte Ollama (local) et OpenAI API.
"""
import os
import json
import requests
from typing import List, Dict, Any, Optional, Literal
from abc import ABC, abstractmethod


# Configuration par défaut
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "gemma3:4b-it-qat"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

# Prompt système par défaut (en anglais)
DEFAULT_SYSTEM_PROMPT = """You are a legal expert assistant specialized in Swiss law.
You answer questions based ONLY on the provided reference documents.
If the information is not in the documents, clearly state that you don't have enough information.
Respond in the same language as the question (French, German, or Italian).
Be precise and cite relevant sources using their document numbers."""


class BaseLLMGenerator(ABC):
    """Classe de base abstraite pour les générateurs LLM."""
    
    @abstractmethod
    def generate(
        self,
        question: str,
        context_documents: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        include_sources: bool = True
    ) -> str:
        """Génère une réponse."""
        pass
    
    @abstractmethod
    def generate_stream(
        self,
        question: str,
        context_documents: List[Dict[str, Any]],
        system_prompt: Optional[str] = None
    ):
        """Génère une réponse en streaming."""
        pass
    
    def _format_context(
        self,
        documents: List[Dict[str, Any]],
        include_metadata: bool = True
    ) -> str:
        """
        Formate les documents de contexte pour le prompt.
        
        Args:
            documents: Liste des documents récupérés
            include_metadata: Inclure les métadonnées (date, numéro d'affaire, etc.)
        
        Returns:
            Contexte formaté en texte
        """
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            metadata = doc.get('metadata', {})
            
            # En-tête du document
            header_parts = [f"[Document {i}]"]
            if include_metadata:
                if metadata.get('case_number'):
                    header_parts.append(f"Case: {metadata['case_number']}")
                if metadata.get('date'):
                    header_parts.append(f"Date: {metadata['date']}")
                if metadata.get('title'):
                    header_parts.append(f"Title: {metadata['title']}")
            
            header = " | ".join(header_parts)
            
            # Texte du document (préférer le texte original si disponible)
            text = metadata.get('original_text', doc.get('document', ''))
            
            # Score de pertinence
            score_info = ""
            if 'rerank_score' in doc:
                score_info = f" (relevance: {doc['rerank_score']:.3f})"
            
            context_parts.append(f"{header}{score_info}\n{text}\n")
        
        return "\n---\n".join(context_parts)
    
    def _build_prompt(
        self,
        question: str,
        context: str,
        system_prompt: str,
        include_sources: bool = True
    ) -> str:
        """Construit le prompt complet."""
        source_instruction = ""
        if include_sources:
            source_instruction = "\n\nIndicate which document numbers you used for your answer."
        
        return f"""{system_prompt}

REFERENCE DOCUMENTS:
{context}

QUESTION: {question}
{source_instruction}

ANSWER:"""


class OllamaGenerator(BaseLLMGenerator):
    """
    Générateur de réponses utilisant Ollama (LLM local).
    
    Exemple d'utilisation:
        ```python
        from src.rag.llm import OllamaGenerator
        
        generator = OllamaGenerator(model="gemma3:4b-it-qat")
        response = generator.generate(
            question="What is the penalty for theft?",
            context_documents=[...]
        )
        ```
    """
    
    def __init__(
        self,
        model: str = DEFAULT_OLLAMA_MODEL,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ):
        """
        Initialise le générateur Ollama.
        
        Args:
            model: Nom du modèle Ollama à utiliser
            ollama_url: URL de l'API Ollama
            temperature: Température pour la génération (0.0 = déterministe, 1.0 = créatif)
            max_tokens: Nombre maximum de tokens à générer
        """
        self.model = model
        self.ollama_url = ollama_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Vérifier la connexion à Ollama
        if not self._check_connection():
            print(f"⚠ Warning: Ollama is not accessible at {ollama_url}")
        else:
            print(f"✓ Connected to Ollama ({model})")
    
    def _check_connection(self) -> bool:
        """Vérifie si Ollama est accessible."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def generate(
        self,
        question: str,
        context_documents: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        include_sources: bool = True
    ) -> str:
        """
        Génère une réponse à la question en utilisant le contexte fourni.
        
        Args:
            question: La question de l'utilisateur
            context_documents: Documents de contexte récupérés par la recherche
            system_prompt: Prompt système personnalisé (None = prompt par défaut)
            include_sources: Demander au LLM d'inclure les sources dans la réponse
        
        Returns:
            Réponse générée par le LLM
        """
        if system_prompt is None:
            system_prompt = DEFAULT_SYSTEM_PROMPT
        
        context = self._format_context(context_documents)
        prompt = self._build_prompt(question, context, system_prompt, include_sources)
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            }
        }
        
        try:
            print(f"  [LLM] Generating response with {self.model}...")
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=180
            )
            response.raise_for_status()
            
            result = response.json()
            answer = result.get("response", "").strip()
            
            if not answer:
                return "Sorry, I couldn't generate a response."
            
            print(f"  [LLM] ✓ Response generated ({len(answer)} characters)")
            return answer
            
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.ollama_url}. "
                "Make sure Ollama is running."
            )
        except requests.exceptions.Timeout:
            raise TimeoutError(
                f"Timeout while generating. Model {self.model} might be too slow."
            )
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                raise ValueError(
                    f"Model '{self.model}' not found. "
                    f"Use 'ollama pull {self.model}' to download it."
                )
            raise
    
    def generate_stream(
        self,
        question: str,
        context_documents: List[Dict[str, Any]],
        system_prompt: Optional[str] = None
    ):
        """Génère une réponse en streaming."""
        if system_prompt is None:
            system_prompt = DEFAULT_SYSTEM_PROMPT
        
        context = self._format_context(context_documents)
        prompt = self._build_prompt(question, context, system_prompt, include_sources=False)
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            }
        }
        
        try:
            with requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                stream=True,
                timeout=180
            ) as response:
                response.raise_for_status()
                
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        token = data.get("response", "")
                        if token:
                            yield token
                        if data.get("done", False):
                            break
                            
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.ollama_url}."
            )


class OpenAIGenerator(BaseLLMGenerator):
    """
    Générateur de réponses utilisant l'API OpenAI.
    
    Exemple d'utilisation:
        ```python
        from src.rag.llm import OpenAIGenerator
        
        generator = OpenAIGenerator(model="gpt-4o-mini")
        response = generator.generate(
            question="What is the penalty for theft?",
            context_documents=[...]
        )
        ```
    """
    
    def __init__(
        self,
        model: str = DEFAULT_OPENAI_MODEL,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ):
        """
        Initialise le générateur OpenAI.
        
        Args:
            model: Nom du modèle OpenAI à utiliser (gpt-4o, gpt-4o-mini, etc.)
            api_key: Clé API OpenAI (si None, utilise OPENAI_API_KEY env var)
            api_base: URL de base de l'API (pour utiliser un endpoint compatible)
            temperature: Température pour la génération
            max_tokens: Nombre maximum de tokens à générer
        """
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.api_base = api_base or "https://api.openai.com/v1"
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        if not self.api_key:
            print("⚠ Warning: No OpenAI API key found. Set OPENAI_API_KEY environment variable.")
        else:
            print(f"✓ OpenAI API configured ({model})")
    
    def _get_headers(self) -> Dict[str, str]:
        """Retourne les headers pour l'API."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def generate(
        self,
        question: str,
        context_documents: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        include_sources: bool = True
    ) -> str:
        """
        Génère une réponse via l'API OpenAI.
        
        Args:
            question: La question de l'utilisateur
            context_documents: Documents de contexte
            system_prompt: Prompt système personnalisé
            include_sources: Inclure les sources dans la réponse
        
        Returns:
            Réponse générée
        """
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
        
        if system_prompt is None:
            system_prompt = DEFAULT_SYSTEM_PROMPT
        
        context = self._format_context(context_documents)
        
        # Construire le message utilisateur
        source_instruction = ""
        if include_sources:
            source_instruction = "\n\nIndicate which document numbers you used for your answer."
        
        user_message = f"""REFERENCE DOCUMENTS:
{context}

QUESTION: {question}
{source_instruction}"""
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        try:
            print(f"  [LLM] Generating response with {self.model}...")
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=self._get_headers(),
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            
            result = response.json()
            answer = result["choices"][0]["message"]["content"].strip()
            
            if not answer:
                return "Sorry, I couldn't generate a response."
            
            print(f"  [LLM] ✓ Response generated ({len(answer)} characters)")
            return answer
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                raise ValueError("Invalid OpenAI API key.")
            elif e.response.status_code == 429:
                raise RuntimeError("OpenAI rate limit exceeded. Please wait and try again.")
            elif e.response.status_code == 404:
                raise ValueError(f"Model '{self.model}' not found or not accessible.")
            raise
        except requests.exceptions.ConnectionError:
            raise ConnectionError("Cannot connect to OpenAI API.")
        except requests.exceptions.Timeout:
            raise TimeoutError("OpenAI API request timed out.")
    
    def generate_stream(
        self,
        question: str,
        context_documents: List[Dict[str, Any]],
        system_prompt: Optional[str] = None
    ):
        """Génère une réponse en streaming via OpenAI."""
        if not self.api_key:
            raise ValueError("OpenAI API key is required.")
        
        if system_prompt is None:
            system_prompt = DEFAULT_SYSTEM_PROMPT
        
        context = self._format_context(context_documents)
        
        user_message = f"""REFERENCE DOCUMENTS:
{context}

QUESTION: {question}"""
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True
        }
        
        try:
            with requests.post(
                f"{self.api_base}/chat/completions",
                headers=self._get_headers(),
                json=payload,
                stream=True,
                timeout=120
            ) as response:
                response.raise_for_status()
                
                for line in response.iter_lines():
                    if line:
                        line_text = line.decode('utf-8')
                        if line_text.startswith("data: "):
                            data_str = line_text[6:]
                            if data_str == "[DONE]":
                                break
                            data = json.loads(data_str)
                            delta = data.get("choices", [{}])[0].get("delta", {})
                            token = delta.get("content", "")
                            if token:
                                yield token
                                
        except requests.exceptions.ConnectionError:
            raise ConnectionError("Cannot connect to OpenAI API.")


# Alias pour compatibilité avec l'ancien code
LLMGenerator = OllamaGenerator


def create_generator(
    backend: Literal["ollama", "openai"] = "ollama",
    model: Optional[str] = None,
    **kwargs
) -> BaseLLMGenerator:
    """
    Crée un générateur LLM selon le backend choisi.
    
    Args:
        backend: "ollama" pour local, "openai" pour OpenAI API
        model: Nom du modèle (si None, utilise le défaut du backend)
        **kwargs: Arguments additionnels passés au constructeur
    
    Returns:
        Instance de générateur LLM
    
    Examples:
        ```python
        # Ollama (local)
        gen = create_generator("ollama", model="gemma3:4b-it-qat")
        
        # OpenAI
        gen = create_generator("openai", model="gpt-4o-mini")
        
        # OpenAI avec clé explicite
        gen = create_generator("openai", model="gpt-4o", api_key="sk-...")
        ```
    """
    if backend == "ollama":
        model = model or DEFAULT_OLLAMA_MODEL
        return OllamaGenerator(model=model, **kwargs)
    elif backend == "openai":
        model = model or DEFAULT_OPENAI_MODEL
        return OpenAIGenerator(model=model, **kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'ollama' or 'openai'.")

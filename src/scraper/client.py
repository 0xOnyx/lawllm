"""
Client HTTP pour l'API Entscheidsuche avec gestion du rate limiting.
"""
import asyncio
import aiohttp
import json
from typing import Optional, Dict, Any, List
from pathlib import Path

from .config import BASE_URL, DOCS_URL, ScraperConfig


class EntscheidsucheClient:
    """
    Client HTTP pour interagir avec l'API Entscheidsuche.
    
    Gère automatiquement le rate limiting et les sessions HTTP.
    """
    
    def __init__(self, config: ScraperConfig):
        self.config = config
        self._session: Optional[aiohttp.ClientSession] = None
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._request_times = []  # Liste des temps des dernières requêtes pour rate limiting global
        self._rate_limiter = asyncio.Lock()
        
    async def __aenter__(self):
        await self._init_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._close_session()
        
    async def _init_session(self):
        """Initialise la session HTTP."""
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
            self._semaphore = asyncio.Semaphore(self.config.max_concurrent)
            
    async def _close_session(self):
        """Ferme la session HTTP."""
        if self._session:
            await self._session.close()
            self._session = None
            
    async def _rate_limited_request(self, url: str) -> Optional[str]:
        """
        Effectue une requête avec rate limiting optimisé.
        
        Args:
            url: URL à requêter
            
        Returns:
            Contenu de la réponse en texte, ou None en cas d'erreur
        """
        async with self._semaphore:
            # Rate limiting global optimisé
            async with self._rate_limiter:
                current_time = asyncio.get_event_loop().time()
                min_interval = 1.0 / self.config.rate_limit
                
                # Nettoyer les temps de requêtes anciennes (garder seulement la dernière seconde)
                self._request_times = [t for t in self._request_times if current_time - t < 1.0]
                
                # Si on a atteint la limite, attendre
                if len(self._request_times) >= self.config.rate_limit:
                    oldest_time = min(self._request_times)
                    wait_time = 1.0 - (current_time - oldest_time)
                    if wait_time > 0:
                        await asyncio.sleep(wait_time)
                        current_time = asyncio.get_event_loop().time()
                        # Nettoyer à nouveau après l'attente
                        self._request_times = [t for t in self._request_times if current_time - t < 1.0]
                
                # Enregistrer le temps de cette requête
                self._request_times.append(current_time)
            
            try:
                async with self._session.get(url) as response:
                    if response.status == 200:
                        return await response.text()
                    elif response.status == 404:
                        return None
                    else:
                        print(f"Erreur {response.status} pour {url}")
                        return None
            except Exception as e:
                print(f"Erreur requête {url}: {e}")
                return None
    
    async def _rate_limited_request_bytes(self, url: str) -> Optional[bytes]:
        """
        Effectue une requête avec rate limiting optimisé (retourne bytes).
        
        Args:
            url: URL à requêter
            
        Returns:
            Contenu de la réponse en bytes, ou None en cas d'erreur
        """
        async with self._semaphore:
            # Rate limiting global optimisé (même logique que pour les requêtes texte)
            async with self._rate_limiter:
                current_time = asyncio.get_event_loop().time()
                min_interval = 1.0 / self.config.rate_limit
                
                # Nettoyer les temps de requêtes anciennes (garder seulement la dernière seconde)
                self._request_times = [t for t in self._request_times if current_time - t < 1.0]
                
                # Si on a atteint la limite, attendre
                if len(self._request_times) >= self.config.rate_limit:
                    oldest_time = min(self._request_times)
                    wait_time = 1.0 - (current_time - oldest_time)
                    if wait_time > 0:
                        await asyncio.sleep(wait_time)
                        current_time = asyncio.get_event_loop().time()
                        # Nettoyer à nouveau après l'attente
                        self._request_times = [t for t in self._request_times if current_time - t < 1.0]
                
                # Enregistrer le temps de cette requête
                self._request_times.append(current_time)
            
            try:
                async with self._session.get(url) as response:
                    if response.status == 200:
                        return await response.read()
                    elif response.status == 404:
                        return None
                    else:
                        print(f"Erreur {response.status} pour {url}")
                        return None
            except Exception as e:
                print(f"Erreur requête PDF {url}: {e}")
                return None
    
    async def get_spider_status(self) -> Dict[str, Any]:
        """Récupère le statut de tous les spiders."""
        await self._init_session()
        url = f"{BASE_URL}/status"
        content = await self._rate_limited_request(url)
        if content:
            return {"raw": content}
        return {}
    
    async def get_jobs_file(self, spider: str) -> Optional[Dict]:
        """
        Récupère le dernier fichier Jobs d'un spider.
        Le fichier Jobs contient la liste complète de tous les documents.
        
        Args:
            spider: Nom du spider (ex: CH_BGer)
            
        Returns:
            Données JSON du fichier Jobs, ou None en cas d'erreur
        """
        await self._init_session()
        url = f"{DOCS_URL}/Jobs/{spider}/last"
        content = await self._rate_limited_request(url)
        if content:
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                print(f"Erreur parsing JSON pour {url}")
        return None
    
    async def get_index_file(self, spider: str) -> Optional[Dict]:
        """
        Récupère le dernier fichier Index d'un spider.
        Le fichier Index contient seulement les mises à jour récentes.
        
        Args:
            spider: Nom du spider (ex: CH_BGer)
            
        Returns:
            Données JSON du fichier Index, ou None en cas d'erreur
        """
        await self._init_session()
        url = f"{DOCS_URL}/Index/{spider}/last"
        content = await self._rate_limited_request(url)
        if content:
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                print(f"Erreur parsing JSON pour {url}")
        return None
    
    async def list_documents(self, spider: str) -> List[str]:
        """
        Liste tous les documents d'un spider via le fichier Jobs.
        
        Args:
            spider: Nom du spider
            
        Returns:
            Liste des chemins des fichiers JSON
        """
        jobs = await self.get_jobs_file(spider)
        if not jobs:
            return []
        
        def extract_json_paths(obj, paths=None):
            """Extrait récursivement les chemins JSON."""
            if paths is None:
                paths = []
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(key, str) and key.endswith('.json'):
                        paths.append(key)
                    extract_json_paths(value, paths)
            elif isinstance(obj, list):
                for item in obj:
                    extract_json_paths(item, paths)
            return paths
        
        return extract_json_paths(jobs, [])
    
    async def fetch_json(self, json_path: str) -> Optional[Dict]:
        """
        Récupère un fichier JSON depuis l'API.
        
        Args:
            json_path: Chemin relatif du fichier JSON
            
        Returns:
            Données JSON parsées, ou None en cas d'erreur
        """
        await self._init_session()
        
        if not json_path.startswith('http'):
            url = f"{DOCS_URL}/{json_path}"
        else:
            url = json_path
            
        content = await self._rate_limited_request(url)
        if content:
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                print(f"Erreur parsing JSON pour {json_path}")
        return None
    
    async def fetch_html(self, html_path: str) -> Optional[str]:
        """
        Récupère le contenu HTML d'un document.
        
        Args:
            html_path: Chemin relatif du fichier HTML
            
        Returns:
            Contenu HTML en texte, ou None en cas d'erreur
        """
        await self._init_session()
        
        if not html_path.startswith('http'):
            url = f"{DOCS_URL}/{html_path}"
        else:
            url = html_path
            
        return await self._rate_limited_request(url)
    
    async def fetch_pdf(self, pdf_path: str) -> Optional[bytes]:
        """
        Récupère le contenu PDF d'un document (binaire).
        
        Args:
            pdf_path: Chemin relatif du fichier PDF
            
        Returns:
            Contenu binaire du PDF, ou None en cas d'erreur
        """
        await self._init_session()
        
        if not pdf_path.startswith('http'):
            url = f"{DOCS_URL}/{pdf_path}"
        else:
            url = pdf_path
        
        return await self._rate_limited_request_bytes(url)


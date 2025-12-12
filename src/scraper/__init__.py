"""
Module de scraping pour l'API Entscheidsuche.

Ce module permet de télécharger les jugements suisses depuis entscheidsuche.ch
de manière efficace et respectueuse du serveur.
"""
from typing import List, Optional, AsyncGenerator
from tqdm.asyncio import tqdm

from .config import ScraperConfig, BASE_URL, DOCS_URL
from .client import EntscheidsucheClient
from .extractors import TextExtractor
from .storage import DocumentStorage
from .normalizer import MetadataNormalizer
from .spider_registry import fetch_spiders_from_api, get_spiders_sync, clear_cache
from ..models import DocumentMetadata, ScrapedDocument
import re


def sanitize_filename(filename: str) -> str:
    """
    Nettoie un nom de fichier pour qu'il soit compatible avec tous les systèmes de fichiers.
    
    Remplace les caractères invalides par des underscores.
    
    Args:
        filename: Nom de fichier à nettoyer
        
    Returns:
        Nom de fichier nettoyé
    """
    # Caractères invalides pour les noms de fichiers Windows/Linux
    # < > : " / \ | ? *
    invalid_chars = r'[<>:"/\\|?*]'
    # Remplacer par underscore
    sanitized = re.sub(invalid_chars, '_', filename)
    # Supprimer les espaces en début/fin
    sanitized = sanitized.strip()
    # Remplacer les espaces multiples par un seul underscore
    sanitized = re.sub(r'\s+', '_', sanitized)
    return sanitized


class EntscheidsucheScraper:
    """
    Scraper principal pour l'API Entscheidsuche.
    
    Exemple d'utilisation:
        ```python
        from src.scraper import EntscheidsucheScraper, ScraperConfig
        
        config = ScraperConfig(output_dir="data/raw", rate_limit=5)
        async with EntscheidsucheScraper(config) as scraper:
            async for doc in scraper.fetch_spider("CH_BGer", max_docs=100):
                print(f"Document: {doc.id}")
        ```
    """
    
    def __init__(self, config: Optional[ScraperConfig] = None):
        self.config = config or ScraperConfig()
        self._client: Optional[EntscheidsucheClient] = None
        self._storage = DocumentStorage(self.config)
        self._extractor = TextExtractor()
        self._normalizer = MetadataNormalizer()
        
    async def __aenter__(self):
        self._client = EntscheidsucheClient(self.config)
        await self._client.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.__aexit__(exc_type, exc_val, exc_tb)
    
    async def fetch_document_metadata(self, json_path: str) -> Optional[DocumentMetadata]:
        """
        Récupère les métadonnées d'un document depuis son fichier JSON.
        
        Args:
            json_path: Chemin du fichier JSON
            
        Returns:
            Métadonnées du document, ou None en cas d'erreur
        """
        data = await self._client.fetch_json(json_path)
        if not data:
            return None
            
        try:
            normalized_data = self._normalizer.normalize(data)
            return DocumentMetadata(**normalized_data)
        except Exception as e:
            print(f"Erreur parsing métadonnées {json_path}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def fetch_full_document(self, json_path: str) -> Optional[ScrapedDocument]:
        """
        Récupère un document complet (métadonnées + contenu).
        
        Gère à la fois les documents HTML et PDF.
        
        Args:
            json_path: Chemin du fichier JSON de métadonnées
            
        Returns:
            Document complet, ou None en cas d'erreur
        """
        # Récupérer les métadonnées
        metadata = await self.fetch_document_metadata(json_path)
        if not metadata:
            return None
        
        content = ""
        content_url = None
        
        # Essayer d'abord HTML, puis PDF
        if metadata.html:
            html = await self._client.fetch_html(metadata.html)
            if html:
                content = self._extractor.extract_from_html(html)
                content_url = f"{DOCS_URL}/{metadata.html}"
        
        # Si pas de contenu HTML, essayer PDF
        if not content and metadata.pdf:
            pdf_bytes = await self._client.fetch_pdf(metadata.pdf)
            if pdf_bytes:
                content = self._extractor.extract_from_pdf(pdf_bytes)
                content_url = f"{DOCS_URL}/{metadata.pdf}"
        
        if not content:
            print(f"Aucun contenu trouvé pour {metadata.signatur}")
            return None
        
        # Extraire le titre
        title = None
        if metadata.Kopfzeile:
            title = metadata.Kopfzeile.get(metadata.lang) or list(metadata.Kopfzeile.values())[0]
        
        # Extraire le résumé
        abstract = None
        if metadata.Abstract:
            abstract = metadata.Abstract.get(metadata.lang) or list(metadata.Abstract.values())[0]
        
        # Déterminer l'URL principale (HTML prioritaire, sinon PDF, sinon JSON)
        entscheidsuche_url = content_url or (f"{DOCS_URL}/{json_path}")
        
        # Générer l'ID combiné: Signatur|Num (avec nettoyage des espaces autour des slashes)
        doc_id = metadata.signatur
        if metadata.num:
            # Nettoyer les espaces autour des slashes: "HC / 2025 / 430" -> "HC/2025/430"
            cleaned_num = metadata.num.replace(" / ", "/").replace("/ ", "/").replace(" /", "/")
            doc_id = f"{metadata.signatur}|{cleaned_num}"
        
        # Nettoyer l'ID pour qu'il soit compatible avec les noms de fichiers
        doc_id = sanitize_filename(doc_id)
        
        return ScrapedDocument(
            id=doc_id,
            spider=metadata.spider,
            language=metadata.lang,
            date=metadata.date,
            case_number=metadata.num,
            title=title,
            abstract=abstract,
            content=content,
            source_url=metadata.url,
            entscheidsuche_url=entscheidsuche_url
        )
    
    async def fetch_spider(
        self, 
        spider: str, 
        max_docs: Optional[int] = None,
        language: Optional[str] = None
    ) -> AsyncGenerator[ScrapedDocument, None]:
        """
        Récupère tous les documents d'un spider.
        
        Args:
            spider: Nom du spider (ex: CH_BGer)
            max_docs: Nombre maximum de documents à récupérer
            language: Filtrer par langue (de, fr, it)
            
        Yields:
            ScrapedDocument: Documents récupérés
        """
        print(f"Récupération de la liste des documents pour {spider}...")
        json_paths = await self._client.list_documents(spider)
        
        if not json_paths:
            print(f"Aucun document trouvé pour {spider}")
            return
        
        if max_docs:
            json_paths = json_paths[:max_docs]
        
        print(f"Téléchargement de {len(json_paths)} documents...")
        
        for json_path in tqdm(json_paths, desc=f"Scraping {spider}"):
            doc = await self.fetch_full_document(json_path)
            if doc:
                if language and doc.language != language:
                    continue
                yield doc
    
    async def fetch_spiders(
        self,
        spiders: List[str],
        max_docs_per_spider: Optional[int] = None,
        language: Optional[str] = None
    ) -> AsyncGenerator[ScrapedDocument, None]:
        """
        Récupère les documents de plusieurs spiders.
        
        Args:
            spiders: Liste des noms de spiders
            max_docs_per_spider: Nombre maximum de documents par spider
            language: Filtrer par langue (de, fr, it)
            
        Yields:
            ScrapedDocument: Documents récupérés
        """
        # Charger la liste des spiders depuis l'API pour les descriptions
        available_spiders = await fetch_spiders_from_api()
        
        for spider in spiders:
            description = available_spiders.get(spider, 'Inconnu')
            print(f"\n=== {spider} ({description}) ===")
            async for doc in self.fetch_spider(spider, max_docs_per_spider, language):
                yield doc
    
    async def save_document(self, doc: ScrapedDocument, output_dir: Optional[str] = None):
        """
        Sauvegarde un document localement.
        
        Args:
            doc: Document à sauvegarder
            output_dir: Répertoire de sortie (optionnel)
            
        Returns:
            Chemin du fichier sauvegardé
        """
        return await self._storage.save_document(doc, output_dir)


# Fonctions utilitaires pour compatibilité ascendante
async def scrape_spider(
    spider: str,
    output_dir: str = "data/raw",
    max_docs: Optional[int] = None,
    language: Optional[str] = None
) -> List[ScrapedDocument]:
    """
    Fonction utilitaire pour scraper un spider.
    
    Exemple:
        ```python
        documents = await scrape_spider("CH_BGer", max_docs=100, language="fr")
        ```
    """
    config = ScraperConfig(output_dir=output_dir)
    documents = []
    
    async with EntscheidsucheScraper(config) as scraper:
        async for doc in scraper.fetch_spider(spider, max_docs, language):
            await scraper.save_document(doc)
            documents.append(doc)
    
    return documents


def list_available_spiders() -> dict:
    """
    Retourne la liste des spiders disponibles depuis l'API.
    
    Returns:
        Dict[str, str]: Dictionnaire {spider_id: description}
    """
    return get_spiders_sync()


# Exports publics
__all__ = [
    'EntscheidsucheScraper',
    'ScraperConfig',
    'scrape_spider',
    'list_available_spiders',
    'fetch_spiders_from_api',
    'get_spiders_sync',
    'clear_cache',
    'DocumentMetadata',
    'ScrapedDocument',
]


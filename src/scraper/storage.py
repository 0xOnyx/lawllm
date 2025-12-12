"""
Gestion du stockage des documents scrapés.
"""
import aiofiles
from pathlib import Path
from typing import Optional

from ..models import ScrapedDocument
from .config import ScraperConfig


class DocumentStorage:
    """Gère la sauvegarde des documents scrapés."""
    
    def __init__(self, config: ScraperConfig):
        self.config = config
        
    async def save_document(
        self, 
        doc: ScrapedDocument, 
        output_dir: Optional[str] = None
    ) -> Path:
        """
        Sauvegarde un document localement.
        
        Args:
            doc: Document à sauvegarder
            output_dir: Répertoire de sortie (utilise config.output_dir si None)
            
        Returns:
            Chemin du fichier sauvegardé
        """
        output_dir = output_dir or self.config.output_dir
        spider_dir = Path(output_dir) / doc.spider
        spider_dir.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder en JSON
        if self.config.save_json:
            filepath = spider_dir / f"{doc.id}.json"
            async with aiofiles.open(filepath, 'w', encoding='utf-8') as f:
                await f.write(doc.model_dump_json(indent=2))
            return filepath
        
        return spider_dir / f"{doc.id}.json"


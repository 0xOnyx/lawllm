"""
Configuration pour le scraper Entscheidsuche.
"""
from dataclasses import dataclass
from typing import Optional


# Configuration par défaut
BASE_URL = "https://entscheidsuche.ch"
DOCS_URL = f"{BASE_URL}/docs"
FACETTEN_URL = f"{DOCS_URL}/Facetten_alle.json"
DEFAULT_RATE_LIMIT = 5  # requêtes par seconde max
DEFAULT_TIMEOUT = 30  # secondes


@dataclass
class ScraperConfig:
    """Configuration du scraper."""
    output_dir: str = "data/raw"
    rate_limit: int = DEFAULT_RATE_LIMIT
    timeout: int = DEFAULT_TIMEOUT
    max_concurrent: int = 10
    save_html: bool = True
    save_json: bool = True
    
    def __post_init__(self):
        """Valide la configuration après initialisation."""
        if self.rate_limit <= 0:
            raise ValueError("rate_limit doit être > 0")
        if self.timeout <= 0:
            raise ValueError("timeout doit être > 0")
        if self.max_concurrent <= 0:
            raise ValueError("max_concurrent doit être > 0")


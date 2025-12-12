"""
Tests pour le scraper (nécessite des mocks pour les appels HTTP).
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.scraper import EntscheidsucheScraper, ScraperConfig
from src.models import ScrapedDocument


class TestScraperConfig:
    """Tests pour ScraperConfig."""
    
    def test_default_config(self):
        """Test configuration par défaut."""
        config = ScraperConfig()
        assert config.output_dir == "data/raw"
        assert config.rate_limit == 5
        assert config.timeout == 30
        assert config.max_concurrent == 10
    
    def test_custom_config(self):
        """Test configuration personnalisée."""
        config = ScraperConfig(
            output_dir="custom/data",
            rate_limit=10,
            timeout=60
        )
        assert config.output_dir == "custom/data"
        assert config.rate_limit == 10
        assert config.timeout == 60
    
    def test_invalid_rate_limit(self):
        """Test validation rate_limit."""
        with pytest.raises(ValueError):
            ScraperConfig(rate_limit=0)
    
    def test_invalid_timeout(self):
        """Test validation timeout."""
        with pytest.raises(ValueError):
            ScraperConfig(timeout=-1)


class TestTextExtractor:
    """Tests pour l'extraction de texte."""
    
    def test_extract_from_html(self):
        """Test extraction depuis HTML."""
        from src.scraper.extractors import TextExtractor
        
        html = """
        <html>
            <head><title>Test</title></head>
            <body>
                <h1>Titre</h1>
                <p>Paragraphe 1</p>
                <p>Paragraphe 2</p>
            </body>
        </html>
        """
        
        text = TextExtractor.extract_from_html(html)
        assert "Titre" in text
        assert "Paragraphe 1" in text
        assert "Paragraphe 2" in text
        # Les balises HTML ne doivent pas être présentes
        assert "<html>" not in text
        assert "<p>" not in text


class TestMetadataNormalizer:
    """Tests pour la normalisation des métadonnées."""
    
    def test_normalize_field_names(self):
        """Test normalisation des noms de champs."""
        from src.scraper.normalizer import MetadataNormalizer
        
        data = {
            "Signatur": "CH_BGer_123",
            "Spider": "CH_BGer",
            "Lang": "fr",
            "Date": "2024-03-15"
        }
        
        normalized = MetadataNormalizer.normalize(data)
        assert normalized["signatur"] == "CH_BGer_123"
        assert normalized["spider"] == "CH_BGer"
        assert normalized["lang"] == "fr"
        assert normalized["date"] == "2024-03-15"
    
    def test_normalize_multilingual_fields(self):
        """Test normalisation des champs multilingues."""
        from src.scraper.normalizer import MetadataNormalizer
        
        data = {
            "signatur": "CH_BGer_123",
            "spider": "CH_BGer",
            "Kopfzeile": [
                {"Sprachen": ["fr"], "Text": "Arrêt français"},
                {"Sprachen": ["de"], "Text": "Urteil deutsch"}
            ]
        }
        
        normalized = MetadataNormalizer.normalize(data)
        assert isinstance(normalized["Kopfzeile"], dict)
        assert normalized["Kopfzeile"]["fr"] == "Arrêt français"
        assert normalized["Kopfzeile"]["de"] == "Urteil deutsch"


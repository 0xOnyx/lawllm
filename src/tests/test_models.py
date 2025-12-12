"""
Tests pour les modèles de données.
"""
import pytest
from datetime import date
from src.models import DocumentMetadata, ScrapedDocument


class TestDocumentMetadata:
    """Tests pour DocumentMetadata."""
    
    def test_create_minimal_metadata(self):
        """Test création avec champs minimaux."""
        metadata = DocumentMetadata(
            signatur="CH_BGer_6B_123_2024",
            spider="CH_BGer"
        )
        assert metadata.signatur == "CH_BGer_6B_123_2024"
        assert metadata.spider == "CH_BGer"
        assert metadata.lang == "de"  # valeur par défaut
    
    def test_create_full_metadata(self):
        """Test création avec tous les champs."""
        metadata = DocumentMetadata(
            signatur="CH_BGer_6B_123_2024",
            spider="CH_BGer",
            lang="fr",
            date="2024-03-15",
            num="6B_123/2024",
            pdf="CH_BGer/6B_123_2024.pdf",
            html="CH_BGer/6B_123_2024.html",
            url="https://www.bger.ch/...",
            Kopfzeile={"fr": "Arrêt du 15 mars 2024"},
            Meta={"fr": "Métadonnées"},
            Abstract={"fr": "Résumé du jugement"}
        )
        assert metadata.lang == "fr"
        assert metadata.date == "2024-03-15"
        assert metadata.Kopfzeile["fr"] == "Arrêt du 15 mars 2024"


class TestScrapedDocument:
    """Tests pour ScrapedDocument."""
    
    def test_create_scraped_document(self):
        """Test création d'un document scrapé."""
        doc = ScrapedDocument(
            id="CH_BGer_6B_123_2024",
            spider="CH_BGer",
            language="fr",
            content="Le Tribunal fédéral a jugé que...",
            entscheidsuche_url="https://entscheidsuche.ch/docs/CH_BGer/6B_123_2024.html"
        )
        assert doc.id == "CH_BGer_6B_123_2024"
        assert doc.spider == "CH_BGer"
        assert doc.language == "fr"
        assert len(doc.content) > 0
    
    def test_scraped_document_optional_fields(self):
        """Test avec tous les champs optionnels."""
        doc = ScrapedDocument(
            id="CH_BGer_6B_123_2024",
            spider="CH_BGer",
            language="fr",
            date="2024-03-15",
            case_number="6B_123/2024",
            title="Arrêt du 15 mars 2024",
            abstract="Résumé",
            content="Contenu...",
            source_url="https://www.bger.ch/...",
            entscheidsuche_url="https://entscheidsuche.ch/docs/CH_BGer/6B_123_2024.html"
        )
        assert doc.date == "2024-03-15"
        assert doc.case_number == "6B_123/2024"
        assert doc.title == "Arrêt du 15 mars 2024"


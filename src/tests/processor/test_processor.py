"""
Tests pour le module processor (découpage par pages).
"""
import pytest
from src.models import ScrapedDocument, TextChunk
from src.rag.processor import chunk_document_by_pages, CHARS_PER_PAGE, PAGES_PER_CHUNK


@pytest.fixture
def sample_document():
    """Crée un document de test avec du texte."""
    # Créer un texte d'environ 10 pages (27000 caractères)
    text = "Premier paragraphe avec plusieurs phrases. " * 100
    text += "\n\n" + "Deuxième paragraphe avec plus de contenu. " * 100
    text += "\n\n" + "Troisième paragraphe pour tester le découpage. " * 100
    
    return ScrapedDocument(
        id="TEST_001",
        spider="CH_BGer",
        language="fr",
        date="2024-01-15",
        case_number="6B_123/2024",
        title="Test Document",
        content=text * 10,  # Environ 10 pages
        entscheidsuche_url="https://example.com/test"
    )


@pytest.fixture
def small_document():
    """Crée un petit document de test (moins d'une page)."""
    return ScrapedDocument(
        id="SMALL_001",
        spider="CH_BGer",
        language="fr",
        date="2024-01-15",
        case_number="6B_123/2024",
        title="Petit Document",
        content="Petit texte de test avec quelques phrases.",
        entscheidsuche_url="https://example.com/small"
    )


@pytest.fixture
def empty_document():
    """Crée un document vide."""
    return ScrapedDocument(
        id="EMPTY_001",
        spider="CH_BGer",
        language="fr",
        content="",
        entscheidsuche_url="https://example.com/empty"
    )


def test_chunk_document_by_pages(sample_document):
    """Test le découpage d'un document par pages."""
    chunks = chunk_document_by_pages(sample_document)
    
    assert len(chunks) > 0
    assert all(isinstance(chunk, TextChunk) for chunk in chunks)
    
    # Vérifier que les métadonnées sont préservées
    first_chunk = chunks[0]
    assert first_chunk.document_id == sample_document.id
    assert first_chunk.spider == sample_document.spider
    assert first_chunk.language == sample_document.language
    assert first_chunk.date == sample_document.date
    assert first_chunk.case_number == sample_document.case_number
    assert first_chunk.title == sample_document.title
    assert first_chunk.entscheidsuche_url == sample_document.entscheidsuche_url


def test_chunk_sizes(sample_document):
    """Test que les chunks ont une taille approximative de 3 pages."""
    chunks = chunk_document_by_pages(sample_document, pages_per_chunk=3)
    
    # Chaque chunk devrait faire environ 3 pages (8100 caractères)
    # On accepte une tolérance de ±20%
    expected_size = 3 * CHARS_PER_PAGE
    min_size = int(expected_size * 0.8)
    max_size = int(expected_size * 1.2)
    
    for chunk in chunks[:-1]:  # Exclure le dernier chunk qui peut être plus petit
        chunk_size = len(chunk.text)
        assert min_size <= chunk_size <= max_size, \
            f"Chunk de taille {chunk_size} hors de la plage attendue [{min_size}, {max_size}]"


def test_chunk_indices(sample_document):
    """Test que les indices des chunks sont séquentiels."""
    chunks = chunk_document_by_pages(sample_document)
    
    # Vérifier que les IDs sont uniques
    chunk_ids = [chunk.id for chunk in chunks]
    assert len(chunk_ids) == len(set(chunk_ids))
    
    # Vérifier que les indices sont séquentiels
    chunk_indices = [chunk.chunk_index for chunk in chunks]
    assert chunk_indices == list(range(len(chunks))), \
        f"Les indices ne sont pas séquentiels: {chunk_indices}"


def test_empty_document(empty_document):
    """Test avec un document vide."""
    chunks = chunk_document_by_pages(empty_document)
    assert len(chunks) == 0


def test_small_document(small_document):
    """Test avec un document petit (moins d'une page)."""
    chunks = chunk_document_by_pages(small_document)
    
    # Un petit document devrait quand même créer au moins un chunk
    assert len(chunks) >= 1
    assert chunks[0].text == small_document.content.strip()


def test_custom_pages_per_chunk(sample_document):
    """Test avec un nombre personnalisé de pages par chunk."""
    # Test avec 5 pages par chunk
    chunks_5_pages = chunk_document_by_pages(sample_document, pages_per_chunk=5)
    
    # Test avec 1 page par chunk
    chunks_1_page = chunk_document_by_pages(sample_document, pages_per_chunk=1)
    
    # Avec plus de pages par chunk, on devrait avoir moins de chunks
    assert len(chunks_5_pages) < len(chunks_1_page)
    
    # Vérifier que les chunks de 5 pages sont plus grands
    if chunks_5_pages and chunks_1_page:
        avg_size_5 = sum(len(c.text) for c in chunks_5_pages) / len(chunks_5_pages)
        avg_size_1 = sum(len(c.text) for c in chunks_1_page) / len(chunks_1_page)
        assert avg_size_5 > avg_size_1


def test_chunk_text_not_empty(sample_document):
    """Test que tous les chunks contiennent du texte non vide."""
    chunks = chunk_document_by_pages(sample_document)
    
    for chunk in chunks:
        assert chunk.text.strip() != "", f"Chunk {chunk.id} est vide"


def test_chunk_covers_full_document(sample_document):
    """Test que tous les chunks couvrent l'intégralité du document."""
    chunks = chunk_document_by_pages(sample_document)
    
    # Reconstruire le texte à partir des chunks
    reconstructed = "".join(chunk.text for chunk in chunks)
    
    # Le texte reconstruit devrait être similaire au texte original
    # (on accepte des différences mineures dues aux espaces)
    original_clean = sample_document.content.replace(" ", "").replace("\n", "")
    reconstructed_clean = reconstructed.replace(" ", "").replace("\n", "")
    
    # Vérifier que la longueur est similaire (tolérance de 5%)
    len_diff = abs(len(original_clean) - len(reconstructed_clean))
    assert len_diff / len(original_clean) < 0.05, \
        f"Différence trop importante: {len_diff} caractères"


def test_chunk_ids_format(sample_document):
    """Test que les IDs des chunks suivent le bon format."""
    chunks = chunk_document_by_pages(sample_document)
    
    for chunk in chunks:
        expected_id = f"{sample_document.id}_chunk_{chunk.chunk_index}"
        assert chunk.id == expected_id, \
            f"ID incorrect: {chunk.id} au lieu de {expected_id}"

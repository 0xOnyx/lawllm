"""
Tests pour le module vector_store.
"""
import pytest
import tempfile
import shutil
from pathlib import Path

from src.models import SummarizedChunk, TextChunk
from src.rag.vector_store import VectorStore, DEFAULT_EMBEDDING_MODEL


@pytest.fixture
def temp_db_dir():
    """Crée un répertoire temporaire pour les tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_summarized_chunks():
    """Crée des SummarizedChunk de test."""
    chunks = []
    for i in range(3):
        text_chunk = TextChunk(
            id=f"TEST_CHUNK_{i}",
            text=f"Texte original du chunk {i} avec du contenu juridique.",
            document_id="TEST_DOC_001",
            chunk_index=i,
            spider="CH_BGer",
            language="fr",
            date="2024-01-15",
            case_number="6B_123/2024",
            title="Test Document",
            entscheidsuche_url="https://example.com/test"
        )
        summarized = SummarizedChunk.from_text_chunk(
            text_chunk,
            f"Résumé juridique du chunk {i} avec points clés."
        )
        chunks.append(summarized)
    return chunks


@pytest.fixture
def vector_store(temp_db_dir):
    """Crée un VectorStore de test."""
    # Utiliser un modèle plus petit pour les tests (si disponible)
    # Sinon utiliser le modèle par défaut
    try:
        store = VectorStore(
            collection_name="test_collection",
            db_path=temp_db_dir,
            embedding_model=DEFAULT_EMBEDDING_MODEL
        )
        yield store
        # Nettoyage
        try:
            store.client.delete_collection(name="test_collection")
        except Exception:
            pass
    except Exception as e:
        pytest.skip(f"Impossible de créer VectorStore: {e}")


def test_vector_store_initialization(vector_store):
    """Test l'initialisation du VectorStore."""
    assert vector_store is not None
    assert vector_store.collection_name == "test_collection"
    assert vector_store.embedding_model is not None


def test_generate_embeddings(vector_store):
    """Test la génération d'embeddings."""
    texts = [
        "Premier texte juridique à encoder.",
        "Deuxième texte avec du contenu différent."
    ]
    
    embeddings = vector_store.generate_embeddings(texts, show_progress=False)
    
    assert len(embeddings) == 2
    assert len(embeddings[0]) > 0
    assert isinstance(embeddings[0][0], float)
    
    # Vérifier que les embeddings ont la bonne dimension
    expected_dim = vector_store.embedding_model.get_sentence_embedding_dimension()
    assert len(embeddings[0]) == expected_dim


def test_index_summarized_chunks(vector_store, sample_summarized_chunks):
    """Test l'indexation de SummarizedChunk."""
    count = vector_store.index_summarized_chunks(
        sample_summarized_chunks,
        batch_size=2,
        show_progress=False
    )
    
    assert count == len(sample_summarized_chunks)
    
    # Vérifier que les documents sont bien indexés
    stats = vector_store.get_collection_stats()
    assert stats['total_documents'] == len(sample_summarized_chunks)


def test_search(vector_store, sample_summarized_chunks):
    """Test la recherche sémantique."""
    # Indexer d'abord
    vector_store.index_summarized_chunks(
        sample_summarized_chunks,
        show_progress=False
    )
    
    # Rechercher
    results = vector_store.search("juridique", n_results=2)
    
    assert len(results) > 0
    assert len(results) <= 2
    
    # Vérifier la structure des résultats
    result = results[0]
    assert 'id' in result
    assert 'document' in result
    assert 'metadata' in result
    assert 'distance' in result or result['distance'] is None


def test_search_with_filter(vector_store, sample_summarized_chunks):
    """Test la recherche avec filtres."""
    # Indexer
    vector_store.index_summarized_chunks(
        sample_summarized_chunks,
        show_progress=False
    )
    
    # Rechercher avec filtre
    results = vector_store.search(
        "juridique",
        n_results=5,
        filter_metadata={"spider": "CH_BGer"}
    )
    
    # Vérifier que tous les résultats respectent le filtre
    for result in results:
        assert result['metadata']['spider'] == "CH_BGer"


def test_get_collection_stats(vector_store, sample_summarized_chunks):
    """Test la récupération des statistiques."""
    vector_store.index_summarized_chunks(
        sample_summarized_chunks,
        show_progress=False
    )
    
    stats = vector_store.get_collection_stats()
    
    assert 'collection_name' in stats
    assert 'total_documents' in stats
    assert 'embedding_model' in stats
    assert 'embedding_dimension' in stats
    assert stats['total_documents'] == len(sample_summarized_chunks)


def test_reset_collection(vector_store, sample_summarized_chunks):
    """Test la réinitialisation de la collection."""
    # Indexer
    vector_store.index_summarized_chunks(
        sample_summarized_chunks,
        show_progress=False
    )
    
    assert vector_store.collection.count() == len(sample_summarized_chunks)
    
    # Réinitialiser
    vector_store.reset_collection()
    
    assert vector_store.collection.count() == 0


def test_index_empty_list(vector_store):
    """Test l'indexation d'une liste vide."""
    count = vector_store.index_summarized_chunks([])
    assert count == 0


def test_search_empty_collection(vector_store):
    """Test la recherche dans une collection vide."""
    results = vector_store.search("test query")
    assert len(results) == 0


def test_metadata_structure(vector_store, sample_summarized_chunks):
    """Test que les métadonnées sont correctement structurées."""
    chunk = sample_summarized_chunks[0]
    metadata = chunk.to_chromadb_metadata()
    
    # Vérifier que toutes les clés nécessaires sont présentes
    required_keys = [
        'original_chunk_id',
        'document_id',
        'chunk_index',
        'spider',
        'language',
        'date',
        'case_number',
        'title',
        'entscheidsuche_url',
        'original_text_length',
        'summary_length'
    ]
    
    for key in required_keys:
        assert key in metadata, f"Clé manquante: {key}"


def test_batch_indexing(vector_store, sample_summarized_chunks):
    """Test l'indexation par batch."""
    # Créer plus de chunks pour tester le batch
    all_chunks = sample_summarized_chunks * 5  # 15 chunks au total
    
    count = vector_store.index_summarized_chunks(
        all_chunks,
        batch_size=5,
        show_progress=False
    )
    
    assert count == len(all_chunks)
    stats = vector_store.get_collection_stats()
    assert stats['total_documents'] == len(all_chunks)


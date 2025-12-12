"""
Configuration pytest pour les tests.
"""
import pytest
import asyncio
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def temp_dir():
    """Crée un répertoire temporaire pour les tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def event_loop():
    """Crée un event loop pour les tests async."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


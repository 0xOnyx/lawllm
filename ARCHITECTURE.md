# Architecture du projet LawLLM

## Vue d'ensemble

Le projet est organisé en modules séparés pour faciliter la maintenance, les tests et l'extension future avec des couches RAG.

## Structure des modules

### `src/models.py`
**Responsabilité** : Définition des modèles de données avec Pydantic
- `DocumentMetadata` : Métadonnées d'un document depuis l'API
- `ScrapedDocument` : Document complet scrapé et traité
- `JobDocument` / `JobsFile` : Structures pour les fichiers Jobs

### `src/scraper/`

#### `config.py`
**Responsabilité** : Configuration centralisée
- `ScraperConfig` : Dataclass avec validation
- Constantes (URLs, limites par défaut)

#### `client.py`
**Responsabilité** : Communication HTTP avec l'API
- `EntscheidsucheClient` : Client async avec rate limiting
- Gestion des sessions HTTP
- Méthodes pour récupérer JSON, HTML, PDF

#### `extractors.py`
**Responsabilité** : Extraction de texte depuis différents formats
- `TextExtractor.extract_from_html()` : Extraction depuis HTML
- `TextExtractor.extract_from_pdf()` : Extraction depuis PDF

#### `storage.py`
**Responsabilité** : Sauvegarde des documents
- `DocumentStorage` : Gère la sauvegarde locale des documents

#### `normalizer.py`
**Responsabilité** : Normalisation des données de l'API
- `MetadataNormalizer` : Convertit les données brutes en format standardisé

#### `spider_registry.py`
**Responsabilité** : Gestion du registre des spiders
- `fetch_spiders_from_api()` : Récupère la liste des spiders (async)
- `get_spiders_sync()` : Version synchrone
- Cache global pour éviter les requêtes répétées

#### `__init__.py`
**Responsabilité** : API principale du module scraper
- `EntscheidsucheScraper` : Classe principale orchestrant tous les composants
- Fonctions utilitaires (`scrape_spider`, `list_available_spiders`)

## Flux de données

```
1. Utilisateur → EntscheidsucheScraper.fetch_spider()
   ↓
2. EntscheidsucheClient.list_documents() → Liste des JSON paths
   ↓
3. Pour chaque document:
   a. EntscheidsucheClient.fetch_json() → Métadonnées brutes
   b. MetadataNormalizer.normalize() → Métadonnées normalisées
   c. DocumentMetadata(**normalized) → Modèle validé
   ↓
4. Récupération du contenu:
   a. EntscheidsucheClient.fetch_html() OU
   b. EntscheidsucheClient.fetch_pdf()
   ↓
5. TextExtractor.extract_from_html/pdf() → Texte extrait
   ↓
6. ScrapedDocument(...) → Document complet
   ↓
7. DocumentStorage.save_document() → Sauvegarde locale
```

## Principes de design

### Séparation des responsabilités
Chaque module a une responsabilité unique et claire :
- **Client** : Communication réseau
- **Extractors** : Traitement de contenu
- **Storage** : Persistance
- **Normalizer** : Transformation de données
- **Config** : Configuration

### Testabilité
Chaque composant peut être testé indépendamment :
- Les dépendances sont injectées (config, client)
- Les méthodes sont statiques quand possible
- Les interfaces sont claires et documentées

### Extensibilité
La structure facilite l'ajout de nouvelles fonctionnalités :
- Nouveau format de document ? → Ajouter méthode dans `extractors.py`
- Nouveau stockage ? → Implémenter interface dans `storage.py`
- Nouvelle source ? → Créer nouveau module similaire à `scraper/`

## Prochaines couches à ajouter

### `src/embeddings/`
- Génération d'embeddings avec sentence-transformers
- Cache des embeddings
- Batch processing

### `src/rag/`
- Indexation vectorielle (ChromaDB)
- Recherche sémantique
- RAG pipeline complet

### `src/api/`
- API REST (FastAPI)
- Endpoints de recherche
- Documentation automatique

## Tests

Les tests sont organisés par module :
- `test_models.py` : Tests des modèles Pydantic
- `test_scraper.py` : Tests du scraper (avec mocks)
- `conftest.py` : Fixtures partagées

## Configuration

- `pyproject.toml` : Configuration du projet Python
- `pytest.ini` : Configuration des tests
- `requirements.txt` : Dépendances


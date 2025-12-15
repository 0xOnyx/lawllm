# LawLLM - RAG pour jugements suisses

Un système de RAG (Retrieval-Augmented Generation) pour rechercher et analyser les jugements judiciaires suisses depuis [entscheidsuche.ch](https://entscheidsuche.ch).

## 🎯 Fonctionnalités

- **Scraping modulaire** : Téléchargement efficace des jugements depuis l'API Entscheidsuche
- **Extraction de texte** : Support HTML et PDF avec extraction automatique
- **Architecture modulaire** : Code organisé et facilement extensible
- **Tests intégrés** : Structure de tests pour garantir la qualité
- **Prêt pour RAG** : Structure prête pour l'ajout de couches d'embeddings et de recherche vectorielle

## 📁 Structure du projet

```
lawllm/
├── src/
│   ├── __init__.py          # Point d'entrée principal
│   ├── models.py            # Modèles de données (Pydantic)
│   ├── scraper/             # Module de scraping
│   │   ├── __init__.py      # API principale du scraper
│   │   ├── config.py        # Configuration
│   │   ├── client.py        # Client HTTP avec rate limiting
│   │   ├── extractors.py    # Extraction de texte (HTML/PDF)
│   │   ├── storage.py        # Sauvegarde des documents
│   │   ├── normalizer.py    # Normalisation des métadonnées
│   │   └── spider_registry.py # Gestion des spiders
│   └── tests/               # Tests unitaires
│       ├── test_models.py
│       ├── test_scraper.py
│       └── conftest.py
├── data/                    # Données scrapées (créé automatiquement)
├── requirements.txt        # Dépendances Python
└── README.md
```

## 🚀 Installation

### Installation standard

```bash
# Cloner le repository
git clone <url>
cd lawllm

# Installer les dépendances
pip install -r requirements.txt
```

### Installation en mode développement (recommandé)

Pour que les imports fonctionnent correctement depuis n'importe où :

```bash
# Installer le package en mode développement
pip install -e .

# Ou avec les dépendances de développement
pip install -e ".[dev]"
```

Cela permet d'importer `src` depuis n'importe quel script du projet.

## 💻 Utilisation

### Scraper un spider

```python
import asyncio
from src import scrape_spider

# Scraper 100 documents du Tribunal fédéral en français
async def main():
    documents = await scrape_spider(
        spider="CH_BGer",
        max_docs=100,
        language="fr"
    )
    print(f"Scrapé {len(documents)} documents")

asyncio.run(main())
```

### Utiliser le scraper avec configuration personnalisée

```python
import asyncio
from src import EntscheidsucheScraper, ScraperConfig

async def main():
    # Configuration personnalisée
    config = ScraperConfig(
        output_dir="data/custom",
        rate_limit=10,  # 10 requêtes/seconde
        max_concurrent=20
    )
    
    async with EntscheidsucheScraper(config) as scraper:
        async for doc in scraper.fetch_spider("CH_BGer", max_docs=50):
            print(f"Document: {doc.id}")
            print(f"Titre: {doc.title}")
            print(f"Contenu: {len(doc.content)} caractères")
            # Le document est automatiquement sauvegardé

asyncio.run(main())
```

### Lister les spiders disponibles

```python
from src import list_available_spiders

spiders = list_available_spiders()
for spider_id, description in spiders.items():
    print(f"{spider_id}: {description}")
```

### Scraper plusieurs spiders

```python
import asyncio
from src import EntscheidsucheScraper, ScraperConfig

async def main():
    config = ScraperConfig(output_dir="data/raw")
    
    async with EntscheidsucheScraper(config) as scraper:
        spiders = ["CH_BGer", "CH_BGE"]
        async for doc in scraper.fetch_spiders(
            spiders=spiders,
            max_docs_per_spider=10,
            language="fr"
        ):
            print(f"{doc.spider}: {doc.id}")

asyncio.run(main())
```

### Pipeline complète d'indexation dans ChromaDB

Le script `main.py` fournit une interface en ligne de commande complète pour ajouter des documents dans ChromaDB. **Par défaut, il fait du scraping depuis l'API Entscheidsuche**, puis chunking, résumé et indexation.

#### Mode par défaut : Scraping complet

Sans aucune option, le script scrape tous les spiders disponibles :

```bash
python main.py
```

#### Scraper des spiders spécifiques

Pour scraper uniquement certains spiders (régions/tribunaux) :

```bash
python main.py --spiders CH_BGer VD_FindInfo
```

#### Scraper uniquement les nouveaux documents

Pour ne télécharger que les documents qui n'existent pas déjà :

```bash
python main.py --only-new
```

#### Limiter le nombre de documents

Pour limiter le nombre de documents par spider :

```bash
python main.py --spiders CH_BGer --max-docs 50
```

#### Filtrer par langue

Pour ne scraper que les documents dans une langue spécifique :

```bash
python main.py --language fr
```

#### Indexer depuis des résumés existants (sans scraping)

Si vous avez déjà des résumés dans `data/summaries` :

```bash
python main.py --from-summaries
```

#### Indexer depuis des chunks (avec résumé automatique)

Si vous avez des chunks dans `data/chunks` et que vous voulez les résumer puis les indexer :

```bash
python main.py --from-chunks
```

#### Pipeline depuis les documents déjà scrapés

Pour faire la pipeline sur des documents déjà téléchargés (sans scraping) :

```bash
python main.py --from-documents
```

#### Options avancées

```bash
# Réinitialiser la collection avant l'indexation
python main.py --reset

# Utiliser un modèle Ollama spécifique pour le résumé
python main.py --ollama-model llama3

# Spécifier un chemin de base ChromaDB personnalisé
python main.py --db-path custom_chroma_db

# Ne pas sauvegarder les fichiers intermédiaires
python main.py --no-save-intermediate

# Utiliser le texte original tronqué au lieu de résumer (plus rapide)
python main.py --skip-summarization

# Ajuster la limite de requêtes par seconde
python main.py --rate-limit 10

# Afficher l'aide complète
python main.py --help
```

#### Exemples de workflow complets

**Workflow 1 : Pipeline complète avec scraping (par défaut)**
```bash
# Scrape tous les spiders, puis chunking, résumé et indexation
python main.py
```

**Workflow 2 : Scraper uniquement certains spiders**
```bash
# Scraper uniquement le Tribunal fédéral et le canton de Vaud
python main.py --spiders CH_BGer VD_FindInfo --only-new
```

**Workflow 3 : Utiliser des résumés existants (sans scraping)**
```bash
# Si vous avez déjà des résumés dans data/summaries
python main.py --from-summaries --reset
```

**Workflow 4 : Sans résumé (plus rapide)**
```bash
# Utiliser le texte original tronqué, pas de résumé
python main.py --skip-summarization
```

**Workflow 5 : Scraping incrémental**
```bash
# Ne scraper que les nouveaux documents depuis la dernière exécution
python main.py --only-new --spiders CH_BGer
```

**Workflow 6 : Accélérer le téléchargement**
```bash
# Augmenter le rate limit et le parallélisme pour télécharger plus vite
python main.py --rate-limit 10 --max-concurrent 20

# Configuration agressive (attention aux limites du serveur)
python main.py --rate-limit 20 --max-concurrent 50
```

#### Optimisation des performances

Pour accélérer le téléchargement de grandes quantités de données :

1. **Augmenter le rate limit** : `--rate-limit 10-20` (au lieu de 5 par défaut)
   - Permet plus de requêtes par seconde
   - Attention : respectez les limites du serveur pour éviter les blocages

2. **Augmenter le parallélisme** : `--max-concurrent 20-50` (au lieu de 10 par défaut)
   - Permet plus de requêtes simultanées
   - Améliore l'utilisation de la bande passante

3. **Combiner les deux** :
   ```bash
   python main.py --rate-limit 15 --max-concurrent 30
   ```

**Note** : Le système utilise un rate limiting optimisé avec fenêtre glissante qui permet de mieux utiliser le parallélisme tout en respectant les limites.

## 🧪 Tests

```bash
# Lancer tous les tests
pytest

# Lancer avec couverture
pytest --cov=src

# Lancer un fichier de test spécifique
pytest src/tests/test_models.py
```

## 📦 Architecture modulaire

Le projet est organisé en modules séparés pour faciliter la maintenance et les tests :

- **`config.py`** : Configuration centralisée avec validation
- **`client.py`** : Client HTTP réutilisable avec rate limiting
- **`extractors.py`** : Extraction de texte depuis différents formats
- **`storage.py`** : Gestion du stockage des documents
- **`normalizer.py`** : Normalisation des données de l'API
- **`spider_registry.py`** : Gestion du registre des spiders

Cette architecture facilite :
- L'ajout de nouvelles fonctionnalités (embeddings, RAG, etc.)
- Les tests unitaires de chaque composant
- La maintenance et le débogage

## 🔮 Prochaines étapes

- [ ] Couche d'embeddings avec sentence-transformers
- [ ] Base de données vectorielle (ChromaDB)
- [ ] Interface de recherche RAG
- [ ] API REST pour interroger les documents
- [ ] Interface web

## 📝 Licence

Voir le fichier LICENSE pour plus de détails.

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou une pull request.


python .\main.py --spiders VD_FindInfo --from-documents  --embedding-batch-size 50 --skip-summarization --max-workers 32 --device cuda   
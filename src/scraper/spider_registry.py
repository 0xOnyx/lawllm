"""
Gestion du registre des spiders disponibles depuis l'API Entscheidsuche.
"""
from typing import Dict, Optional
import requests
import aiohttp
from .config import FACETTEN_URL, DEFAULT_TIMEOUT


# Cache global pour les spiders (évite de recharger à chaque appel)
_spiders_cache: Optional[Dict[str, str]] = None


def _parse_spiders_from_data(data: Dict) -> Dict[str, str]:
    """
    Parse la structure JSON de l'API pour extraire les spiders.
    
    Args:
        data: Données JSON de l'API Facetten
        
    Returns:
        Dict[str, str]: Dictionnaire {spider_id: description_fr}
    """
    spiders: Dict[str, str] = {}
    
    for kanton_code, kanton_data in data.items():
        if not isinstance(kanton_data, dict):
            continue
        
        gerichte = kanton_data.get("gerichte", {})
        for gericht_id, gericht_data in gerichte.items():
            if not isinstance(gericht_data, dict):
                continue
            
            # Récupérer la description (préférence: fr > de > it)
            description = (
                gericht_data.get("fr") or 
                gericht_data.get("de") or 
                gericht_data.get("it") or 
                gericht_id
            )
            
            # Les spiders sont dans les "kammern" (chambres)
            kammern = gericht_data.get("kammern", {})
            for kammer_id, kammer_data in kammern.items():
                if isinstance(kammer_data, dict):
                    spider_name = kammer_data.get("spider")
                    if spider_name and spider_name not in spiders:
                        # Utiliser la description du tribunal parent
                        spiders[spider_name] = description
    
    return spiders


async def fetch_spiders_from_api(timeout: int = DEFAULT_TIMEOUT) -> Dict[str, str]:
    """
    Récupère la liste des spiders depuis l'API Entscheidsuche (version async).
    
    L'API fournit un fichier JSON (Facetten_alle.json) contenant la liste 
    complète des juridictions et tribunaux avec leurs noms en allemand, 
    français et italien.
    
    Args:
        timeout: Timeout en secondes pour la requête
        
    Returns:
        Dict[str, str]: Dictionnaire {spider_id: description_fr}
    """
    global _spiders_cache
    
    if _spiders_cache is not None:
        return _spiders_cache
    
    spiders: Dict[str, str] = {}
    
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
            async with session.get(FACETTEN_URL) as response:
                if response.status == 200:
                    data = await response.json()
                    spiders = _parse_spiders_from_data(data)
                    _spiders_cache = spiders
                    print(f"✓ {len(spiders)} spiders chargés depuis l'API")
                else:
                    print(f"Erreur API ({response.status}), utilisation du cache local")
    except Exception as e:
        print(f"Erreur lors de la récupération des spiders: {e}")
    
    return spiders


def get_spiders_sync(timeout: int = DEFAULT_TIMEOUT) -> Dict[str, str]:
    """
    Version synchrone pour récupérer les spiders.
    Utile pour les contextes non-async.
    
    Args:
        timeout: Timeout en secondes pour la requête
        
    Returns:
        Dict[str, str]: Dictionnaire {spider_id: description_fr}
    """
    global _spiders_cache
    
    if _spiders_cache is not None:
        return _spiders_cache
    
    spiders: Dict[str, str] = {}
    
    try:
        response = requests.get(FACETTEN_URL, timeout=timeout)
        if response.status_code == 200:
            data = response.json()
            spiders = _parse_spiders_from_data(data)
            _spiders_cache = spiders
            print(f"✓ {len(spiders)} spiders chargés depuis l'API")
    except Exception as e:
        print(f"Erreur lors de la récupération des spiders: {e}")
    
    return spiders


def clear_cache():
    """Efface le cache des spiders (utile pour les tests)."""
    global _spiders_cache
    _spiders_cache = None


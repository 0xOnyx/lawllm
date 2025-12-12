"""
Normalisation des métadonnées depuis l'API vers les modèles Pydantic.
"""
from typing import Dict, Any


class MetadataNormalizer:
    """Normalise les données JSON de l'API pour correspondre aux modèles."""
    
    @staticmethod
    def normalize(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalise les données JSON pour correspondre au modèle DocumentMetadata.
        
        L'API utilise parfois des noms de champs différents (Signatur vs signatur)
        et des structures différentes (listes vs dicts pour Kopfzeile, Meta, Abstract).
        
        Args:
            data: Données brutes de l'API
            
        Returns:
            Données normalisées
        """
        normalized = {}
        
        # Normaliser les noms de champs (majuscules/minuscules)
        field_mapping = {
            'Signatur': 'signatur',
            'Spider': 'spider',
            'Lang': 'lang',
            'Date': 'date',
            'Datum': 'date',  # Variante allemande
            'Num': 'num',
            'PDF': 'pdf',
            'Html': 'html',
            'HTML': 'html',
            'Url': 'url',
            'URL': 'url',
        }
        
        for key, value in data.items():
            # Ignorer Kopfzeile, Meta, Abstract ici - ils seront traités séparément
            if key in ['Kopfzeile', 'Meta', 'Abstract']:
                continue
                
            # Mapper les noms de champs
            normalized_key = field_mapping.get(key, key.lower())
            
            # Ne pas écraser si la clé existe déjà (éviter les doublons)
            if normalized_key not in normalized:
                normalized[normalized_key] = value
        
        # Normaliser Kopfzeile, Meta, Abstract (peuvent être des listes ou des dicts)
        for field_name in ['Kopfzeile', 'Meta', 'Abstract']:
            value = data.get(field_name)
            if value is not None:
                # Si c'est une liste, convertir en dict
                if isinstance(value, list) and len(value) > 0:
                    result_dict = {}
                    for item in value:
                        if isinstance(item, dict):
                            languages = item.get('Sprachen', [])
                            text = item.get('Text', '')
                            if isinstance(languages, list):
                                for lang in languages:
                                    if lang and text:
                                        result_dict[lang] = text
                            elif languages and text:
                                result_dict[languages] = text
                    normalized[field_name] = result_dict if result_dict else None
                elif isinstance(value, dict):
                    normalized[field_name] = value
                else:
                    normalized[field_name] = None
        
        # Normaliser les champs qui peuvent être des listes ou des dicts
        for field in ['num', 'pdf', 'html', 'url']:
            if field in normalized:
                value = normalized[field]
                # Si c'est une liste, prendre le premier élément
                if isinstance(value, list) and len(value) > 0:
                    normalized[field] = value[0] if isinstance(value[0], str) else str(value[0])
                # Si c'est un dict, extraire la valeur de 'Datei' ou la première valeur
                elif isinstance(value, dict):
                    # Pour PDF et HTML, extraire aussi l'URL originale si elle existe
                    if field in ['pdf', 'html'] and 'URL' in value:
                        if 'url' not in normalized or not normalized['url']:
                            normalized['url'] = value['URL']
                    elif field in ['pdf', 'html'] and 'url' in value:
                        if 'url' not in normalized or not normalized['url']:
                            normalized['url'] = value['url']
                    
                    # Chercher 'Datei' en priorité pour le chemin du fichier
                    if 'Datei' in value:
                        normalized[field] = value['Datei']
                    elif 'datei' in value:
                        normalized[field] = value['datei']
                    else:
                        # Prendre la première valeur qui est une string (mais pas URL)
                        for k, v in value.items():
                            if isinstance(v, str) and k.upper() != 'URL':
                                normalized[field] = v
                                break
                        # Si aucune string trouvée, convertir le dict en string
                        if field in normalized and not isinstance(normalized[field], str):
                            normalized[field] = str(value)
        
        return normalized


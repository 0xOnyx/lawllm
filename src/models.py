"""
Modèles de données pour les jugements suisses.
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
from datetime import date


class DocumentMetadata(BaseModel):
    """Métadonnées d'un document JSON de l'API Entscheidsuche."""
    signatur: str = Field(..., description="Identifiant unique du document")
    spider: str = Field(..., description="Nom du scraper (ex: CH_BGer)")
    lang: str = Field(default="de", description="Langue du document (de, fr, it)")
    date: Optional[str] = Field(default=None, description="Date de la décision")
    num: Optional[str] = Field(default=None, description="Numéro de l'affaire")
    
    # Chemins des fichiers
    pdf: Optional[str] = Field(default=None, description="Chemin du PDF")
    html: Optional[str] = Field(default=None, description="Chemin du HTML")
    url: Optional[str] = Field(default=None, description="URL originale")
    
    # Informations d'affichage (peuvent être multilingues)
    Kopfzeile: Optional[Dict[str, str]] = Field(default=None, description="En-tête/titre")
    Meta: Optional[Dict[str, str]] = Field(default=None, description="Métadonnées courtes")
    Abstract: Optional[Dict[str, str]] = Field(default=None, description="Résumé")


class JobDocument(BaseModel):
    """Document dans un fichier Jobs."""
    path: str
    status: str  # neu, identisch, update, nicht_mehr_da, etc.
    checksum: Optional[str] = None


class JobsFile(BaseModel):
    """Structure d'un fichier Jobs."""
    spider: str
    job: str
    time: str
    documents: List[JobDocument] = Field(default_factory=list)


class ScrapedDocument(BaseModel):
    """Document complet récupéré et traité."""
    id: str = Field(..., description="Identifiant unique (signatur)")
    spider: str
    language: str
    date: Optional[str] = None
    case_number: Optional[str] = None
    title: Optional[str] = None
    abstract: Optional[str] = None
    content: str = Field(..., description="Texte complet extrait")
    source_url: Optional[str] = None
    entscheidsuche_url: str = Field(..., description="URL sur entscheidsuche.ch")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "CH_BGer_6B_123_2024",
                "spider": "CH_BGer",
                "language": "fr",
                "date": "2024-03-15",
                "case_number": "6B_123/2024",
                "title": "Arrêt du 15 mars 2024",
                "abstract": "Résumé du jugement",
                "content": "Le Tribunal fédéral a jugé que...",
                "source_url": "https://www.bger.ch/...",
                "entscheidsuche_url": "https://entscheidsuche.ch/docs/CH_BGer/6B_123_2024.html"
            }
        }


class TextChunk(BaseModel):
    """Chunk de texte avec métadonnées pour le stockage vectoriel."""
    id: str = Field(..., description="Identifiant unique du chunk")
    text: str = Field(..., description="Texte du chunk")
    document_id: str = Field(..., description="ID du document source")
    chunk_index: int = Field(..., description="Index du chunk dans le document")
    spider: str = Field(..., description="Spider source")
    language: str = Field(..., description="Langue du chunk")
    date: Optional[str] = Field(default=None, description="Date du jugement")
    case_number: Optional[str] = Field(default=None, description="Numéro d'affaire")
    title: Optional[str] = Field(default=None, description="Titre du jugement")
    entscheidsuche_url: str = Field(..., description="URL du document source")


class SummarizedChunk(BaseModel):
    """
    Chunk résumé avec métadonnées pour le stockage dans ChromaDB.
    
    Ce modèle contient le résumé du chunk original ainsi que toutes les métadonnées
    nécessaires pour retrouver le texte original et le document source.
    """
    id: str = Field(..., description="Identifiant unique du chunk résumé (même que le chunk original)")
    summary: str = Field(..., description="Résumé du chunk (max 500 caractères)")
    original_chunk_id: str = Field(..., description="ID du chunk original")
    original_text: str = Field(..., description="Texte original du chunk (pour référence)")
    document_id: str = Field(..., description="ID du document source")
    chunk_index: int = Field(..., description="Index du chunk dans le document")
    spider: str = Field(..., description="Spider source")
    language: str = Field(..., description="Langue du chunk")
    date: Optional[str] = Field(default=None, description="Date du jugement")
    case_number: Optional[str] = Field(default=None, description="Numéro d'affaire")
    title: Optional[str] = Field(default=None, description="Titre du jugement")
    entscheidsuche_url: str = Field(..., description="URL du document source")
    
    def to_chromadb_metadata(self) -> dict:
        """
        Convertit le SummarizedChunk en métadonnées pour ChromaDB.
        
        Returns:
            Dictionnaire de métadonnées compatible avec ChromaDB
        """
        return {
            "original_chunk_id": self.original_chunk_id,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "spider": self.spider,
            "language": self.language,
            "date": self.date or "",
            "case_number": self.case_number or "",
            "title": self.title or "",
            "entscheidsuche_url": self.entscheidsuche_url,
            "original_text_length": len(self.original_text),
            "summary_length": len(self.summary),
        }
    
    @classmethod
    def from_text_chunk(cls, chunk: TextChunk, summary: str) -> "SummarizedChunk":
        """
        Crée un SummarizedChunk à partir d'un TextChunk et de son résumé.
        
        Args:
            chunk: Le chunk original
            summary: Le résumé du chunk
            
        Returns:
            SummarizedChunk avec toutes les métadonnées
        """
        return cls(
            id=chunk.id,
            summary=summary,
            original_chunk_id=chunk.id,
            original_text=chunk.text,
            document_id=chunk.document_id,
            chunk_index=chunk.chunk_index,
            spider=chunk.spider,
            language=chunk.language,
            date=chunk.date,
            case_number=chunk.case_number,
            title=chunk.title,
            entscheidsuche_url=chunk.entscheidsuche_url
        )



"""
Extracteurs de texte depuis HTML et PDF.
"""
import re
import io
from typing import Optional
from bs4 import BeautifulSoup


class TextExtractor:
    """Classe de base pour l'extraction de texte."""
    
    @staticmethod
    def extract_from_html(html_content: str) -> str:
        """
        Extrait le texte propre depuis le HTML.
        
        Args:
            html_content: Contenu HTML brut
            
        Returns:
            Texte extrait et nettoyé
        """
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Supprimer les scripts et styles
        for element in soup(['script', 'style', 'nav', 'header', 'footer']):
            element.decompose()
        
        # Extraire le texte
        text = soup.get_text(separator='\n')
        
        # Nettoyer les espaces multiples
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        
        return text.strip()
    
    @staticmethod
    def extract_from_pdf(pdf_content: bytes) -> str:
        """
        Extrait le texte depuis un PDF.
        
        Args:
            pdf_content: Contenu binaire du PDF
            
        Returns:
            Texte extrait du PDF
        """
        try:
            import pdfplumber
            
            text_parts = []
            with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
            
            text = '\n\n'.join(text_parts)
            
            # Nettoyer les espaces multiples
            text = re.sub(r'\n\s*\n', '\n\n', text)
            text = re.sub(r' +', ' ', text)
            
            return text.strip()
        except ImportError:
            raise ImportError(
                "pdfplumber n'est pas installé. "
                "Installez-le avec: pip install pdfplumber"
            )
        except Exception as e:
            print(f"Erreur lors de l'extraction du texte PDF: {e}")
            return ""


"""
Literature Searcher - Busqueda automatica de fuentes academicas.

Busca en:
- Crossref (gratuito, sin API key)
- PubMed (gratuito, sin API key)

Se ejecuta automaticamente cuando el usuario hace una pregunta.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json
import re
import unicodedata
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET


@dataclass
class LiteratureResult:
    title: str
    authors: str
    year: int
    source: str
    doi: Optional[str]
    url: Optional[str]
    abstract: str
    relevance: str


class LiteratureSearcher:
    """
    Busca literatura academica relacionada con la pregunta del usuario.
    """

    def __init__(self):
        self._cache: List[LiteratureResult] = []

    def auto_search(self, question: str, max_results: int = 3) -> List[LiteratureResult]:
        """
        Busca automaticamente papers relacionados con la pregunta.

        Estrategia:
        1. Crossref con sinonimos academicos + palabras de la pregunta
        2. Si no hay resultados: Crossref solo con las palabras de la pregunta
        3. Si aun no hay resultados: PubMed
        """
        keywords = self._extract_keywords(question)
        query = " ".join(keywords[:8])

        results = []

        # 1) Crossref con la consulta completa
        try:
            results = self.search_crossref(query, max_results=max_results)
        except Exception:
            results = []

        # 2) Fallback: solo las palabras reales de la pregunta
        if not results:
            own_words = [
                t for t in re.findall(r"[a-záéíóúñü]{3,}", question.lower())
                if t not in self._STOPWORDS
            ]
            if own_words:
                try:
                    results = self.search_crossref(
                        " ".join(own_words[:6]), max_results=max_results
                    )
                except Exception:
                    results = []

        # 3) PubMed solo si Crossref no encontro nada
        if not results:
            try:
                results = self.search_pubmed(query, max_results=max_results)
            except Exception:
                results = []

        self._cache.extend(results)
        return results

    # Mapa de conceptos a keywords academicas (coincidencia por PALABRA COMPLETA)
    _CONCEPT_MAP: Dict[str, List[str]] = {
        "genero": ["gender", "student wellbeing"],
        "mujer": ["gender", "women", "student wellbeing"],
        "hombre": ["gender", "men", "student wellbeing"],
        "seccional": ["campus", "university", "student wellbeing"],
        "bogota": ["Colombia", "university", "student wellbeing"],
        "modalidad": ["online education", "student wellbeing"],
        "pregrado": ["undergraduate", "student wellbeing"],
        "posgrado": ["graduate", "student wellbeing"],
        "estrato": ["socioeconomic", "student wellbeing"],
        "edad": ["age", "student wellbeing"],
        "dimension": ["psychological wellbeing", "Ryff"],
        "autoaceptacion": ["self-acceptance", "Ryff", "psychological wellbeing"],
        "autonomia": ["autonomy", "Ryff", "psychological wellbeing"],
        "relaciones": ["positive relationships", "Ryff", "psychological wellbeing"],
        "crecimiento": ["personal growth", "Ryff", "psychological wellbeing"],
        "proposito": ["purpose in life", "Ryff", "psychological wellbeing"],
        "dominio": ["environmental mastery", "Ryff", "psychological wellbeing"],
        "bienestar": ["student wellbeing", "psychological wellbeing"],
        "felicidad": ["student wellbeing", "happiness", "university"],
        "salud mental": ["mental health", "university students"],
        "estres": ["stress", "university students"],
        "ansiedad": ["anxiety", "university students"],
        "depresion": ["depression", "university students"],
        "satisfaccion": ["satisfaction", "university students"],
        "rendimiento": ["academic performance", "student wellbeing"],
        "escala": ["scale", "psychometric"],
        "ryff": ["Ryff", "psychological wellbeing"],
    }

    # Palabras vacias (espanol + ingles) que no aportan a la busqueda
    _STOPWORDS = {
        # espanol
        "dame", "dime", "muestra", "muestrame", "quiero", "necesito",
        "busca", "buscar", "bucame", "buscame", "dame", "sobre", "acerca",
        "referencias", "referencia", "fuentes", "fuente", "citas", "cita",
        "articulos", "articulo", "literatura", "bibliografia", "publicaciones",
        "que", "cual", "cuales", "donde", "como", "cuando", "quien",
        "del", "de", "la", "el", "los", "las", "un", "una", "unos", "unas",
        "y", "o", "u", "en", "para", "por", "con", "sin", "es", "son",
        "hay", "me", "mi", "tu", "se", "al", "lo", "le", "les",
        "mas", "este", "esta", "estos", "estas", "todo", "todos",
        "tema", "temas", "existe", "existentes", "tienes", "tengo",
        "apoyos", "apoyo", "sustento", "marco", "teorico", "teorica",
        "estudio", "estudios", "proyecto", "datos",
        # ingles
        "the", "of", "on", "in", "for", "and", "or", "about", "what",
        "which", "who", "when", "where", "how", "show", "give", "me",
        "my", "is", "are", "was", "were", "there", "their", "this",
        "that", "these", "with", "from", "to", "by", "as", "at",
        "references", "reference", "articles", "article", "papers",
        "paper", "literature", "find", "get",
    }

    def _extract_keywords(self, question: str) -> List[str]:
        """
        Extrae palabras clave relevantes de la pregunta.

        Combina dos fuentes:
        1. Sinonimos academicos del mapa de conceptos (matching por palabra
           completa, para no confundir "edad" dentro de "ansiedad")
        2. Las palabras REALES de la pregunta del usuario, sin palabras vacias
           (para no perder terminos como "Ryff" o "autoaceptacion")
        """
        question_lower = question.lower()

        # Sin acentos, para que "autoaceptacion" casen con "autoaceptación"
        question_flat = "".join(
            c for c in unicodedata.normalize("NFD", question_lower)
            if unicodedata.category(c) != "Mn"
        )

        # 1) Sinonimos academicos del mapa de conceptos
        synonyms: List[str] = []
        for concept, kw_list in self._CONCEPT_MAP.items():
            if re.search(r"\b" + re.escape(concept) + r"\b", question_flat):
                synonyms.extend(kw_list)

        # 2) Palabras reales de la pregunta (sin stopwords, min. 3 letras)
        tokens = re.findall(r"[a-záéíóúñü]{3,}", question_lower)
        own_words = [t for t in tokens if t not in self._STOPWORDS]

        # Combinar: sinonimos academicos primero (mejor recall en Crossref)
        keywords = synonyms + own_words

        # 3) Respaldo si la pregunta no aporto nada util
        if not keywords:
            keywords = ["student wellbeing", "university"]

        # Unico ignorando mayusculas/acentos, manteniendo el primer formato
        seen = set()
        unique: List[str] = []
        for kw in keywords:
            key = "".join(
                c for c in unicodedata.normalize("NFD", kw.lower())
                if unicodedata.category(c) != "Mn"
            )
            if key not in seen:
                seen.add(key)
                unique.append(kw)
        return unique

    def search_crossref(
        self,
        query: str,
        max_results: int = 3,
    ) -> List[LiteratureResult]:
        """Busca en Crossref API (gratuito, sin API key)."""
        try:
            encoded = urllib.parse.quote(query)
            url = f"https://api.crossref.org/works?query={encoded}&rows={max_results}&sort=relevance"
            req = urllib.request.Request(url, headers={"User-Agent": "USTA-BienestarAgent/1.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())

            results = []
            for item in data.get("message", {}).get("items", []):
                title = item.get("title", [""])[0]
                if not title:
                    continue
                authors = ", ".join([
                    f"{a.get('given', '')} {a.get('family', '')}"
                    for a in item.get("author", [])[:3]
                ])
                year_data = item.get("published-print", item.get("published-online", {}))
                year_val = year_data.get("date-parts", [[0]])[0][0] if year_data else 0
                source = item.get("container-title", [""])[0]
                doi = item.get("DOI", None)
                abstract = item.get("abstract", "")[:300]

                results.append(LiteratureResult(
                    title=title[:200],
                    authors=authors[:100],
                    year=year_val,
                    source=source[:100],
                    doi=doi,
                    url=f"https://doi.org/{doi}" if doi else None,
                    abstract=abstract,
                    relevance="Fuente academica",
                ))

            return results

        except Exception:
            return []

    def search_pubmed(
        self,
        query: str,
        max_results: int = 3,
    ) -> List[LiteratureResult]:
        """Busca en PubMed API (gratuito)."""
        try:
            encoded = urllib.parse.quote(query)
            search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&retmax={max_results}&term={encoded}"
            req = urllib.request.Request(search_url)
            with urllib.request.urlopen(req, timeout=5) as resp:
                search_data = resp.read().decode()

            root = ET.fromstring(search_data)
            ids = [id_elem.text for id_elem in root.findall(".//Id")]

            if not ids:
                return []

            fetch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={','.join(ids)}"
            req2 = urllib.request.Request(fetch_url)
            with urllib.request.urlopen(req2, timeout=5) as resp2:
                summary_data = resp2.read().decode()

            root2 = ET.fromstring(summary_data)
            results = []
            for article in root2.findall(".//DocSum"):
                title_elem = article.find(".//Item[@Name='Title']")
                authors_elem = article.find(".//Item[@Name='Authors']")
                source_elem = article.find(".//Item[@Name='Source']")
                pubdate = article.find(".//Item[@Name='PubDate']")
                uid_elem = article.find(".//Id")

                results.append(LiteratureResult(
                    title=title_elem.text if title_elem is not None else "",
                    authors=authors_elem.text if authors_elem is not None else "",
                    year=int(pubdate.text.split()[0]) if pubdate is not None and pubdate.text else 0,
                    source=source_elem.text if source_elem is not None else "PubMed",
                    doi=None,
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{uid_elem.text}" if uid_elem is not None else None,
                    abstract="Ver articulo completo en PubMed",
                    relevance="PubMed - fuente medica",
                ))

            return results

        except Exception:
            return []

    def format_for_response(self, results: List[LiteratureResult]) -> str:
        """Formatea resultados para incluir en la respuesta."""
        if not results:
            return ""

        lines = ["\n--- Fuentes relacionadas encontradas ---"]
        for i, r in enumerate(results[:3], 1):
            year_str = f"({r.year})" if r.year > 0 else ""
            lines.append(f"{i}. {r.title} {year_str}")
            lines.append(f"   Fuente: {r.source} | DOI: {r.doi or 'N/A'}")
            if r.url:
                lines.append(f"   URL: {r.url}")

        return "\n".join(lines)

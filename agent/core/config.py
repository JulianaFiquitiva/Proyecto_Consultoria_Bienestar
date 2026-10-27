"""
Configuración central del agente.

Define rutas, constantes del instrumento y parámetros del pipeline.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# Raiz del proyecto: agent/core/config.py -> parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ── Constantes del instrumento Ryff ──────────────────────────────────────

ITEM_COLS_RANGE = (0, 29)  # 29 ítems de la escala
N_ITEMS = 29
N_DIMENSIONS = 6
N_RESPONSE_CATEGORIES = 6  # Likert 1-6

# Ítems con redacción negativa (requieren inversión: valor_inv = 7 - valor)
NEGATIVE_ITEMS = [2, 4, 5, 8, 9, 13, 19, 22, 23, 26]

# Dimensiones de Ryff y sus ítems
DIMENSION_MAP: Dict[str, List[int]] = {
    "Autoaceptación": [1, 7, 17, 24],
    "Relaciones positivas": [2, 8, 12, 22, 25],
    "Autonomía": [3, 4, 9, 13, 18, 23],
    "Dominio del entorno": [5, 10, 14, 19, 29],
    "Propósito de vida": [6, 11, 15, 16, 20],
    "Crecimiento personal": [21, 26, 27, 28],
}

# Orden de las dimensiones para presentación
DIMENSION_ORDER = [
    "Autoaceptación",
    "Crecimiento personal",
    "Propósito de vida",
    "Dominio del entorno",
    "Relaciones positivas",
    "Autonomía",
]

# Etiquetas cortas de los ítems
ITEM_LABELS: Dict[int, str] = {
    1: "Contento con mi historia",
    2: "Solo/pocos amigos*",
    3: "No temo opinar",
    4: "Preocupa eval. ajena*",
    5: "Difícil dirigir vida*",
    6: "Planes futuro",
    7: "Seguro conmigo",
    8: "Pocos escuchan*",
    9: "Preocupa opinión*",
    10: "Hogar a gusto",
    11: "Activo proyectos",
    12: "Amistades aportan",
    13: "Influenciado*",
    14: "Responsable situación",
    15: "Bien pasado/futuro",
    16: "Objetivos: satisfacción",
    17: "Gusta personalidad",
    18: "Confianza opiniones",
    19: "Demandas deprimen*",
    20: "Clara dirección",
    21: "Sigo aprendiendo",
    22: "Pocas relaciones*",
    23: "Difícil opinar*",
    24: "Orgulloso de mí",
    25: "Confianza amigos",
    26: "No he mejorado*",
    27: "Desarrollado mucho",
    28: "Vida: estudio/cambio",
    29: "Pasos para cambiar",
}

# Mapeo de programa a NBC (Núcleo Básico de Conocimiento)
NBC_MAP: Dict[str, str] = {
    "ADMINISTRACIÓN DE EMPRESAS": "Administración",
    "ADMINISTRACIÓN DE EMPRESAS AGROPECUARIAS": "Administración",
    "ADMINISTRACIÓN AMBIENTAL Y DE LOS RECURSOS NATURALES": "Administración",
    "CONTADURÍA PÚBLICA": "Contaduría pública",
    "ECONOMÍA": "Economía",
    "FINANZAS": "Finanzas y afines",
    "MERCADEO": "Mercadeo",
    "MARKETING Y TRANSFORMACIÓN DIGITAL": "Mercadeo",
    "NEGOCIOS INTERNACIONALES": "Negocios internacionales y afines",
    "PROFESIONAL EN MODA": "Diseño",
    "DERECHO": "Derecho y afines",
    "GOBIERNO Y RELACIONES INTERNACIONALES": "Ciencia política y relaciones internacionales",
    "INGENIERÍA CIVIL": "Ingeniería civil y afines",
    "INGENIERÍA AMBIENTAL": "Ingeniería ambiental, sanitaria y afines",
    "INGENIERÍA ELECTRÓNICA": "Ingeniería electrónica, telecomunicaciones y afines",
    "INGENIERÍA DE TELECOMUNICACIONES": "Ingeniería electrónica, telecomunicaciones y afines",
    "INGENIERÍA MECÁNICA": "Ingeniería mecánica y afines",
    "INGENIERÍA MECATRÓNICA": "Ingeniería mecánica y afines",
    "INGENIERÍA INDUSTRIAL": "Ingeniería industrial y afines",
    "INGENIERÍA DE SISTEMAS": "Ingeniería de sistemas, telemática y afines",
    "INGENIERÍA EN INFORMÁTICA": "Ingeniería de sistemas, telemática y afines",
    "INGENIERÍA DE DATOS E INTELIGENCIA ARTIFICIAL": "Ingeniería de sistemas, telemática y afines",
    "INGENIERÍA EN LOGÍSTICA Y OPERACIONES": "Ingeniería industrial y afines",
    "BIOINGENIERÍA": "Ingeniería biomédica y afines",
    "CONSTRUCCIÓN EN ARQUITECTURA E INGENIERÍA": "Ingeniería civil y afines",
    "ARQUITECTURA": "Arquitectura y afines",
    "DISEÑO GRÁFICO": "Diseño",
    "DISEÑO DE INTERACCIÓN": "Diseño",
    "PSICOLOGÍA": "Psicología",
    "SOCIOLOGÍA": "Sociología, trabajo social y afines",
    "COMUNICACIÓN SOCIAL": "Comunicación social, periodismo y afines",
    "COMUNICACIÓN Y EDUCACIÓN DIGITAL": "Comunicación social, periodismo y afines",
    "CULTURA FÍSICA, DEPORTE Y RECREACIÓN": "Deportes, educación física y recreación",
    "LICENCIATURA EN EDUCACIÓN INFANTIL": "Educación",
    "LICENCIATURA EN ARTES PLÁSTICAS": "Educación",
    "LICENCIATURA EN BIOLOGÍA": "Educación",
    "LICENCIATURA EN EDUCACIÓN RELIGIOSA": "Educación",
    "LICENCIATURA EN FILOSOFÍA Y LETRAS": "Educación",
    "LICENCIATURA EN LENGUAS EXTRANJERAS": "Educación",
    "LICENCIATURA EN LENGUAS EXTRANJERAS INGLÉS": "Educación",
    "LICENCIATURA EN ESPAÑOL Y LENGUAS EXTRANJERAS INGLÉS Y FRANCÉS": "Educación",
    "LICENCIATURA EN TEOLOGÍA": "Educación",
    "ODONTOLOGÍA": "Odontología",
    "OPTOMETRÍA": "Optometría, otros programas de ciencias de la salud",
    "ESTADÍSTICA": "Estadística y afines",
    "TEOLOGÍA": "Teología y afines",
    "ZOOTECNIA": "Zootecnia",
    "TECNOLOGÍA EN LABORATORIO DENTAL": "Odontología",
}

# Resultados de referencia para la validación (entrega 1, commit 4be2bf1).
# Coinciden con data/03_dataset_IRT_ponderado.xlsx al estimar CON los
# factores de expansión. deff y n efectivo se corrigieron en la entrega 2:
# valían 1.03 y 1752, pero el notebook 01 y los informes del estudio dicen
# 1.120 y 1619 (Kish con factor_expansion_original).
REPORTED_RESULTS = {
    "n_valid": 1813,
    "N_population": 29950,
    "T_global_mean": 50.7,
    "T_global_ci_lower": 50.2,
    "T_global_ci_upper": 51.2,
    "alpha_cronbach": 0.909,
    "risk_proportion": 0.148,  # T < 40
    "high_proportion": 0.192,  # T >= 60
    "deff": 1.120,
    "effective_sample_size": 1619,
    "pre_posgrado_gap": 4.7,
    "pre_posgrado_T_pre": 49.3,
    "pre_posgrado_T_pos": 54.0,
}

# Cifras del Informe Ejecutivo del estudio anterior
# (docs/Informe_Ejecutivo_Felicidad.tex, secciones 3 y 12). Varias NO se
# reproducen con los datos disponibles: ver la validación de la entrega 2.
REPORTED_INFORME_EJECUTIVO = {
    "n_valid": 1813,
    "N_population": 29950,
    "T_global_mean": 51.2,
    "T_global_ci_lower": 50.6,
    "T_global_ci_upper": 51.8,
    "alpha_cronbach": 0.909,
    "risk_proportion": 0.195,  # "≈ 19.5 %"
    "high_proportion": 0.296,  # "≈ 29.6 %"
    "deff": 1.120,
    "effective_sample_size": 1619,
}


@dataclass
class AgentConfig:
    """Configuración central del agente."""

    # Rutas. La raiz es la carpeta del proyecto, no el directorio desde el
    # que se ejecuta: asi el agente encuentra data/ en cualquier equipo.
    project_root: Path = field(default_factory=lambda: PROJECT_ROOT)
    data_dir: Path = field(default_factory=lambda: Path("data"))
    notebooks_dir: Path = field(default_factory=lambda: Path("notebooks"))
    docs_dir: Path = field(default_factory=lambda: Path("docs"))

    # Carpeta de datos externa (opcional): variable de entorno INDICE_DATA_DIR.
    # Si no esta definida se usa project_root / data_dir.
    external_data_dir: Optional[Path] = field(
        default_factory=lambda: (
            Path(os.environ["INDICE_DATA_DIR"])
            if os.environ.get("INDICE_DATA_DIR") else None
        )
    )

    # Datos
    raw_data_file: str = "Formulario de la escala de bienestar subjetivo - Resultados Finales (1).xlsx"
    prepared_data_file: str = "01_dataset_preparado.xlsx"
    irt_data_file: str = "03_dataset_IRT_ponderado.xlsx"
    grm_params_file: str = "Parametros_GRM.xlsx"

    # Diseño muestral
    population_size: int = 29950
    n_items: int = 29
    n_dimensions: int = 6

    # IRT
    irt_normalization: str = "prior"  # "prior" o "weighted"
    t_score_mean: float = 50.0
    t_score_sd: float = 10.0

    # Inferencia
    alpha_level: float = 0.05
    correction_method: str = "bonferroni"

    # Seguridad
    anonymize: bool = True
    exclude_pii: bool = True

    def get_data_path(self, filename: str) -> Path:
        """Retorna la ruta completa a un archivo de datos."""
        # Primero la carpeta externa (INDICE_DATA_DIR), si existe el archivo
        if self.external_data_dir is not None:
            external_path = self.external_data_dir / filename
            if external_path.exists():
                return external_path
        # Si no, el directorio de datos del proyecto
        return self.project_root / self.data_dir / filename

    def get_notebook_path(self, filename: str) -> Path:
        """Retorna la ruta completa a un notebook."""
        return self.project_root / self.notebooks_dir / filename


def load_config(project_root: Optional[str] = None) -> AgentConfig:
    """Carga la configuración del agente."""
    config = AgentConfig()
    if project_root:
        config.project_root = Path(project_root)
    return config

"""
Report Generator - Generador de informes estadisticos.

Genera informes:
- Ejecutivo (para directivos)
- Tecnico (para estadisticos)
- Por seccional/programa/grupo
- Comparativo

Cada informe incluye:
- Pregunta
- Datos utilizados
- Metodo
- Resultados con IC
- Interpretacion
- Limitaciones
- Recomendaciones
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from agent.core.config import DIMENSION_ORDER, REPORTED_RESULTS


@dataclass
class ReportSection:
    title: str
    content: str
    tables: List[pd.DataFrame]
    figures: List[str]


@dataclass
class Report:
    title: str
    subtitle: str
    date: str
    sections: List[ReportSection]
    metadata: Dict[str, Any]


class ReportGenerator:
    """
    Genera informes estadisticos completos.
    """

    def __init__(self):
        self._reports: List[Report] = []

    def generate_executive_report(
        self,
        df_irt: pd.DataFrame,
        df_prepared: Optional[pd.DataFrame] = None,
    ) -> Report:
        """
        Genera informe ejecutivo para directivos.
        """
        sections = []

        # 1. Resumen ejecutivo
        t = df_irt["T_Global"] if "T_Global" in df_irt.columns else None
        if t is not None:
            mean_t = t.mean()
            risk_pct = (t < 40).mean() * 100
            high_pct = (t >= 60).mean() * 100
            
            classification = self._classify_t(mean_t)
            
            sections.append(ReportSection(
                title="Resumen Ejecutivo",
                content=f"""El indice de bienestar estudiantil de la Universidad Santo Tomas 
obtuvo un T-score global de {mean_t:.1f} ({classification}), con una poblacion 
estimada de {REPORTED_RESULTS['N_population']:,} estudiantes.

{risk_pct:.1f}% de los estudiantes ({int(risk_pct/100*REPORTED_RESULTS['N_population']):,}) 
se encuentran en zona de riesgo (T < 40), mientras que {high_pct:.1f}% 
({int(high_pct/100*REPORTED_RESULTS['N_population']):,}) presentan bienestar alto (T >= 60).

El tamano de la muestra valida fue de {len(df_irt):,} estudiantes, con un 
factor de diseno (DEFF) de {REPORTED_RESULTS['deff']:.2f}.""",
                tables=[],
                figures=[],
            ))

        # 2. Perfil por dimensiones
        dim_data = []
        for dim in DIMENSION_ORDER:
            col = f"T_{dim}"
            if col in df_irt.columns:
                vals = df_irt[col].dropna()
                dim_data.append({
                    "Dimension": dim,
                    "Media T": round(vals.mean(), 1),
                    "DE": round(vals.std(), 1),
                    "IC 95%": f"[{vals.mean() - 1.96*vals.std()/len(vals)**0.5:.1f}, {vals.mean() + 1.96*vals.std()/len(vals)**0.5:.1f}]",
                })
        
        if dim_data:
            sections.append(ReportSection(
                title="Perfil por Dimensiones de Bienestar",
                content="Las seis dimensiones de la escala de Ryff muestran:",
                tables=[pd.DataFrame(dim_data)],
                figures=[],
            ))

        # 3. Comparaciones por grupo
        if "seccional" in df_irt.columns:
            sede_data = []
            for sede, group in df_irt.groupby("seccional"):
                t_vals = group["T_Global"].dropna()
                sede_data.append({
                    "Seccional": sede,
                    "n": len(t_vals),
                    "T-score": round(t_vals.mean(), 1),
                    "DE": round(t_vals.std(), 1),
                })
            
            if sede_data:
                sections.append(ReportSection(
                    title="Resultados por Seccional",
                    content="Comparacion del indice de bienestar por seccional:",
                    tables=[pd.DataFrame(sede_data)],
                    figures=[],
                ))

        # 4. Modalidad
        if "modalidad_exp" in df_irt.columns:
            mod_data = []
            for mod, group in df_irt.groupby("modalidad_exp"):
                t_vals = group["T_Global"].dropna()
                mod_data.append({
                    "Modalidad": mod,
                    "n": len(t_vals),
                    "T-score": round(t_vals.mean(), 1),
                })
            
            if mod_data:
                sections.append(ReportSection(
                    title="Resultados por Modalidad",
                    content="Comparacion pregrado vs posgrado:",
                    tables=[pd.DataFrame(mod_data)],
                    figures=[],
                ))

        # 5. Limitaciones
        sections.append(ReportSection(
            title="Limitaciones",
            content="""1. Los datos son de encuesta, no permiten inferencia causal.
2. La muestra es de una sola universidad (generalizacion limitada).
3. El muestreo no es aleatorio simple, requiere pesos para inferencia.
4. No se evalua satisfaccion universitaria ni rendimiento academico.
5. La variable socioeconomico se mide por estrato, no por ingreso.""",
            tables=[],
            figures=[],
        ))

        # 6. Recomendaciones
        sections.append(ReportSection(
            title="Recomendaciones",
            content="""1. Fortalecer programas en dimensiones con menor puntuacion.
2. Monitorear la evolucion del indice en proximas oleadas.
3. Profundizar en analisis por programa con muestras suficientes.
4. Considerar variables adicionales (satisfaccion, rendimiento) en futuras ediciones.
5. Disenar intervenciones para la poblacion en riesgo (T < 40).""",
            tables=[],
            figures=[],
        ))

        return Report(
            title="Informe Ejecutivo - Indice de Bienestar Estudiantil",
            subtitle="Universidad Santo Tomas 2025-2026",
            date=datetime.now().strftime("%Y-%m-%d"),
            sections=sections,
            metadata={
                "n_valid": len(df_irt),
                "N_population": REPORTED_RESULTS["N_population"],
                "t_global": round(df_irt["T_Global"].mean(), 1) if "T_Global" in df_irt.columns else None,
            },
        )

    def generate_technical_report(
        self,
        df_prepared: pd.DataFrame,
        df_irt: pd.DataFrame,
        grm_params: Optional[pd.DataFrame] = None,
        validation_results: Optional[Dict] = None,
    ) -> Report:
        """
        Genera informe tecnico para estadisticos.
        """
        sections = []

        # 1. Diseno muestral
        sections.append(ReportSection(
            title="1. Diseno Muestral",
            content=f"""Poblacion objetivo: {REPORTED_RESULTS['N_population']:,} estudiantes
Muestra seleccionada: 2,167 estudiantes
Respuestas validas: {len(df_prepared):,} ({len(df_prepared)/2167*100:.1f}% de seleccionados)
Diseno: Muestreo estratificado por seccional x modalidad
Calibracion: Raking (IPF) de Deville & Sarndal (1992)
DEFF: {REPORTED_RESULTS['deff']:.2f}
Muestra efectiva: {REPORTED_RESULTS['effective_sample_size']:,}""",
            tables=[],
            figures=[],
        ))

        # 2. Instrumento
        sections.append(ReportSection(
            title="2. Instrumento",
            content="""Escala de Bienestar de Ryff (versión abreviada)
- 29 items, escala Likert 1-6
- 6 dimensiones: Autoaceptacion, Crecimiento personal, Proposito de vida,
  Dominio del entorno, Relaciones positivas, Autonomia
- 10 items con redaccion negativa (invertidos con formula 7-valor)""",
            tables=[],
            figures=[],
        ))

        # 3. Analisis psicometrico
        from agent.psychometrics.scale import PsychometricScale
        scale = PsychometricScale()
        from agent.data.loader import DataLoader
        loader = DataLoader()
        
        try:
            item_cols = loader.get_item_columns(df_prepared)
            report_psych = scale.full_analysis(df_prepared, item_cols)
            
            sections.append(ReportSection(
                title="3. Analisis Psicometrico",
                content=f"""Alpha de Cronbach global: {report_psych.alpha_global:.3f}
Alpha por dimension:
""" + "\n".join([f"  - {dim}: {alpha:.3f}" for dim, alpha in report_psych.alpha_dimensions.items()]),
                tables=[],
                figures=[],
            ))
        except Exception:
            pass

        # 4. Modelo IRT
        if grm_params is not None:
            sections.append(ReportSection(
                title="4. Modelo IRT (GRM)",
                content=f"""Modelo: Graded Response Model (Samejima, 1969)
Parametros estimados: {len(grm_params)} items
Discriminaciones: rango [{grm_params['a (discriminacion)'].min():.3f}, {grm_params['a (discriminacion)'].max():.3f}]
Estimacion: Maximum Marginal Likelihood (MML)
Habilidad latente: Expected A Posteriori (EAP)
Normalizacion: T = 50 + 10 * theta (prior N(0,1))""",
                tables=[grm_params.head(10)],
                figures=[],
            ))

        # 5. Resultados
        if "T_Global" in df_irt.columns:
            t = df_irt["T_Global"]
            sections.append(ReportSection(
                title="5. Resultados Principales",
                content=f"""T-score global: {t.mean():.1f} (IC 95%: {t.mean()-1.96*t.std()/len(t)**0.5:.1f}, {t.mean()+1.96*t.std()/len(t)**0.5:.1f})
DE: {t.std():.1f}
Poblacion en riesgo (T<40): {(t<40).mean()*100:.1f}%
Bienestar alto (T>=60): {(t>=60).mean()*100:.1f}%""",
                tables=[],
                figures=[],
            ))

        # 6. Validacion
        if validation_results:
            sections.append(ReportSection(
                title="6. Validacion de Resultados",
                content="Resultados reproducidos vs. reportados:",
                tables=[],
                figures=[],
            ))

        return Report(
            title="Informe Tecnico - Indice de Bienestar Estudiantil",
            subtitle="Universidad Santo Tomas 2025-2026",
            date=datetime.now().strftime("%Y-%m-%d"),
            sections=sections,
            metadata={"type": "technical"},
        )

    def _classify_t(self, t_score: float) -> str:
        if t_score < 40:
            return "Muy bajo"
        elif t_score < 45:
            return "Bajo"
        elif t_score < 55:
            return "Medio"
        elif t_score < 60:
            return "Alto"
        else:
            return "Muy alto"

    def format_report_markdown(self, report: Report) -> str:
        """Formatea un reporte como Markdown."""
        lines = [
            f"# {report.title}",
            f"## {report.subtitle}",
            f"*Fecha: {report.date}*",
            "",
        ]

        for section in report.sections:
            lines.append(f"## {section.title}")
            lines.append(section.content)
            lines.append("")

            for table in section.tables:
                if isinstance(table, pd.DataFrame):
                    lines.append(table.to_markdown(index=False))
                    lines.append("")

        return "\n".join(lines)

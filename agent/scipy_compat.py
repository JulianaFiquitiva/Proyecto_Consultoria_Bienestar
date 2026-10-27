"""
Carga protegida de scipy.stats.

Smart App Control puede bloquear un DLL de scipy sin aviso. Si el import
falla, los modulos estadisticos se siguen pudiendo importar (el agente, el
RAG y el dashboard arrancan) y solo falla, con un mensaje claro, el calculo
que de verdad necesita scipy.

Uso en los modulos:

    from agent.scipy_compat import stats as sp_stats
"""
from typing import Optional


class ScipyNoDisponible(RuntimeError):
    """El calculo pedido necesita scipy y scipy no se pudo cargar."""


class _StatsNoDisponible:
    """Sustituto de scipy.stats: cualquier uso explica por que falla."""

    def __init__(self, error: str):
        self._error = error

    def __getattr__(self, nombre: str):
        raise ScipyNoDisponible(
            f"Este calculo necesita scipy.stats.{nombre}, pero scipy no se "
            f"pudo cargar ({self._error}). Si Smart App Control lo bloquea, "
            f"el resto del agente sigue funcionando."
        )

    def __bool__(self) -> bool:
        return False


try:
    from scipy import stats
    SCIPY_ERROR: Optional[str] = None
except BaseException as exc:            # incluye el SystemExit de SAC
    SCIPY_ERROR = f"{type(exc).__name__}: {exc}"
    stats = _StatsNoDisponible(SCIPY_ERROR)

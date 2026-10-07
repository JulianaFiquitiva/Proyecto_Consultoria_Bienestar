"""
Suite de Demostración con Datos Reales.

Ejecuta casos de uso completos para demostrar que el agente
funciona correctamente con los datos del estudio.

Uso:
    python -m tests.demo_integration
    
Requiere:
    - Datos en C:/Users/Lenovo-Laptop/Documents/IndiceFelicidad/
    - API key configurada (opcional, funciona sin LLM)
"""

import sys
import time
from pathlib import Path

import pandas as pd

# Agregar directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.api_client import LLMClient, LLMConfig, LLMProvider
from agent.core.config import AgentConfig, DIMENSION_ORDER
from agent.core.orchestrator import StatisticalAgent


class DemoSuite:
    """
    Suite de demostración con casos de prueba reales.
    """

    def __init__(self, use_llm: bool = False, provider: str = "openai"):
        self.config = AgentConfig()
        self.results = []
        
        if use_llm:
            self.llm_client = LLMClient(LLMConfig(provider=LLMProvider(provider)))
        else:
            self.llm_client = None
            
        self.agent = StatisticalAgent(llm_client=self.llm_client, config=self.config)

    def load_data(self):
        """Carga los datos del estudio."""
        print("=" * 60)
        print("CARGANDO DATOS DEL ESTUDIO")
        print("=" * 60)
        
        try:
            self.agent.load_data()
            print("[OK] Datos cargados correctamente")
            if self.agent._irt_data is not None:
                print(f"  Dataset IRT: {self.agent._irt_data.shape[0]} registros, {self.agent._irt_data.shape[1]} columnas")
            if self.agent._data is not None:
                print(f"  Dataset preparado: {self.agent._data.shape[0]} registros")
        except Exception as e:
            print(f"[ERROR] Error cargando datos: {e}")
            return False
        return True

    def run_all_demos(self):
        """Ejecuta todas las demostraciones."""
        demos = [
            ("1. Resumen General", self.demo_overview),
            ("2. Comparación por Género", self.demo_gender_comparison),
            ("3. Comparación por Seccional", self.demo_seccional_comparison),
            ("4. Comparación por Modalidad", self.demo_modality_comparison),
            ("5. Perfil por Dimensiones", self.demo_dimension_profile),
            ("6. Caso de Fallo: Variable Inexistente", self.demo_missing_variable),
            ("7. Caso de Fallo: Pregunta Ambigua", self.demo_ambiguous_question),
            ("8. Validación contra Resultados Anteriores", self.demo_validation),
        ]

        print("\n" + "=" * 60)
        print("EJECUTANDO DEMOSTRACIONES")
        print("=" * 60)

        for name, demo_func in demos:
            print(f"\n{'-' * 40}")
            print(f"Demo: {name}")
            print(f"{'-' * 40}")
            
            start_time = time.time()
            try:
                result = demo_func()
                elapsed = time.time() - start_time
                self.results.append({
                    "demo": name,
                    "success": True,
                    "time_ms": round(elapsed * 1000, 2),
                    "result_summary": str(result)[:200] if result else "OK",
                })
                print(f"[OK] Completada en {elapsed*1000:.0f}ms")
            except Exception as e:
                elapsed = time.time() - start_time
                self.results.append({
                    "demo": name,
                    "success": False,
                    "time_ms": round(elapsed * 1000, 2),
                    "error": str(e),
                })
                print(f"[ERROR] Error: {e}")

        self._print_summary()

    def demo_overview(self) -> dict:
        """Demuestra resumen general del bienestar."""
        response = self.agent.ask("Dame un resumen general del bienestar estudiantil")
        print(f"\n  Pregunta: {response.question}")
        print(f"  Tipo: {response.plan.analysis_type.value}")
        if response.success:
            print(f"  Resultado: Media T={response.result.get('media', 'N/A')}")
            print(f"  Interpretación: {response.interpretation[:150]}...")
        else:
            print(f"  Error: {response.error}")
        return response.result

    def demo_gender_comparison(self) -> dict:
        """Demuestra comparación por género."""
        response = self.agent.ask("¿Hay diferencia en bienestar por género?")
        print(f"\n  Pregunta: {response.question}")
        if response.success and "prueba_estadistica" in response.result:
            test = response.result["prueba_estadistica"]
            print(f"  Test: {test.get('test', 'N/A')}")
            print(f"  p-valor: {test.get('p_valor', 'N/A')}")
            print(f"  Significativo: {test.get('significativo', 'N/A')}")
        return response.result

    def demo_seccional_comparison(self) -> dict:
        """Demuestra comparación por seccional."""
        response = self.agent.ask("¿Cómo varía el bienestar entre sedes?")
        print(f"\n  Pregunta: {response.question}")
        if response.success:
            for sede, stats in response.result.items():
                if isinstance(stats, dict) and "media" in stats:
                    print(f"  {sede}: T={stats['media']} (n={stats['n']})")
        return response.result

    def demo_modality_comparison(self) -> dict:
        """Demuestra comparación por modalidad."""
        response = self.agent.ask("¿Difieren pregrado y posgrado en bienestar?")
        print(f"\n  Pregunta: {response.question}")
        if response.success and "prueba_estadistica" in response.result:
            test = response.result["prueba_estadistica"]
            print(f"  Test: {test.get('test', 'N/A')}")
            print(f"  Significativo: {test.get('significativo', 'N/A')}")
        return response.result

    def demo_dimension_profile(self) -> dict:
        """Demuestra perfil por dimensiones."""
        response = self.agent.ask("¿Cuáles son las dimensiones más fuertes y débiles?")
        print(f"\n  Pregunta: {response.question}")
        if response.success:
            for dim, stats in response.result.items():
                if isinstance(stats, dict) and "media" in stats:
                    print(f"  {dim}: T={stats['media']}")
            if "_resumen" in response.result:
                res = response.result["_resumen"]
                print(f"  Más baja: {res.get('dimension_mas_baja', 'N/A')}")
                print(f"  Más alta: {res.get('dimension_mas_alta', 'N/A')}")
        return response.result

    def demo_missing_variable(self) -> dict:
        """Demuestra manejo de variable inexistente."""
        response = self.agent.ask("¿Cuál es el ingreso promedio de los estudiantes?")
        print(f"\n  Pregunta: {response.question}")
        print(f"  Éxito: {response.success}")
        print(f"  Error: {response.error or 'N/A'}")
        print(f"  Interpretación: {response.interpretation[:150]}...")
        return {"success": response.success, "error": response.error}

    def demo_ambiguous_question(self) -> dict:
        """Demuestra manejo de pregunta ambigua."""
        response = self.agent.ask("¿Qué opinan los estudiantes?")
        print(f"\n  Pregunta: {response.question}")
        print(f"  Tipo detectado: {response.plan.analysis_type.value}")
        print(f"  Confianza: {response.plan.confidence}")
        return {"type": response.plan.analysis_type.value, "confidence": response.plan.confidence}

    def demo_validation(self) -> dict:
        """Demuestra validación contra resultados anteriores."""
        response = self.agent.ask("Valida los resultados contra lo reportado")
        print(f"\n  Pregunta: {response.question}")
        if response.success and "resumen" in response.result:
            summary = response.result["resumen"]
            # El summary es un string, no un dict
            print(f"  Resumen:\n{summary[:300]}...")
        return response.result

    def _print_summary(self):
        """Imprime resumen de la demostración."""
        print("\n" + "=" * 60)
        print("RESUMEN DE DEMOSTRACIONES")
        print("=" * 60)
        
        passed = sum(1 for r in self.results if r["success"])
        failed = sum(1 for r in self.results if not r["success"])
        total = len(self.results)
        
        print(f"\nTotal: {total}")
        print(f"Exitosas: {passed}")
        print(f"Fallidas: {failed}")
        
        if self.results:
            avg_time = sum(r["time_ms"] for r in self.results) / len(self.results)
            print(f"Tiempo promedio: {avg_time:.0f}ms")
        
        print("\nDetalle:")
        for r in self.results:
            status = "[OK]" if r["success"] else "[FAIL]"
            print(f"  {status} {r['demo']} ({r['time_ms']:.0f}ms)")
            if not r["success"]:
                print(f"    Error: {r.get('error', 'N/A')}")


def main():
    """Función principal de demostración."""
    print("Agente Estadístico - Índice de Bienestar Estudiantil")
    print("Universidad Santo Tomás")
    print()
    
    # Verificar si se quiere usar LLM
    use_llm = "--llm" in sys.argv
    provider = "openai"
    for arg in sys.argv:
        if arg.startswith("--provider="):
            provider = arg.split("=")[1]
    
    demo = DemoSuite(use_llm=use_llm, provider=provider)
    
    if not demo.load_data():
        print("\nNo se pudieron cargar los datos. Verifica la ruta.")
        return
    
    demo.run_all_demos()


if __name__ == "__main__":
    main()

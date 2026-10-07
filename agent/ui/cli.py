"""
Interfaz de linha de comandos para el Agente Estadistico.
Ejecutar desde la terminal de VS Code:

    python -m agent.ui.cli
    
O directamente:

    python agent/ui/cli.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agent.core.orchestrator import StatisticalAgent


def print_banner():
    print("=" * 60)
    print("  AGENTE ESTADISTICO - Indice de Bienestar USTA")
    print("  Universidad Santo Tomas 2025-2026")
    print("=" * 60)
    print()
    print("  Comandos disponibles:")
    print("    resumen        - Resumen general del bienestar")
    print("    genero         - Comparar bienestar por genero")
    print("    seccional      - Comparar bienestar por seccional")
    print("    modalidad      - Comparar pregrado vs posgrado")
    print("    estrato        - Comparar por estrato socioeconomico")
    print("    dimensiones    - Perfil por dimensiones de Ryff")
    print("    edad           - Correlacion edad-bienestar")
    print("    regresion      - Regresion: que factores predicen bienestar")
    print("    validar        - Validar resultados contra estudio anterior")
    print("    validacion     - Validacion completa del pipeline")
    print("    buscar <query> - Buscar literatura academica (web)")
    print("    corpus <pregunta> - Preguntar al corpus local con cita")
    print("    ayuda          - Mostrar esta ayuda")
    print("    salir          - Salir del agente")
    print()
    print("  Tambien puedes hacer preguntas en lenguaje natural:")
    print('    "Como difiere el bienestar por genero?"')
    print('    "Cual es el T-score de Bogota?"')
    print('    "Que dimension es mas debil?"')
    print()
    print("  Para consultar el corpus documental (20 PDF, con cita):")
    print('    corpus ¿Que dicen los documentos sobre bienestar y sueño?')
    print()


def main():
    print_banner()

    print("Cargando datos...")
    agent = StatisticalAgent()
    agent.load_data()

    if agent._data is None:
        print("ERROR: No se pudieron cargar los datos.")
        print("Verifica que la carpeta 'data/' contenga los archivos .xlsx")
        return

    n_prepared = len(agent._data) if agent._data is not None else 0
    n_irt = len(agent._irt_data) if agent._irt_data is not None else 0
    print(f"Datos cargados: {n_prepared} registros preparados, {n_irt} con T-scores IRT")
    print()

    while True:
        try:
            question = input("Tu pregunta > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nHasta pronto!")
            break

        if not question:
            continue

        if question.lower() in ("salir", "exit", "quit", "q"):
            print("Hasta pronto!")
            break

        if question.lower() in ("ayuda", "help", "h", "?"):
            print_banner()
            continue

        # Mapear comandos rapidos a preguntas
        command_map = {
            "resumen": "Cual es el resumen general del bienestar?",
            "genero": "Como difiere el bienestar por genero?",
            "seccional": "Como varia el bienestar entre seccionales?",
            "modalidad": "Como difiere el bienestar entre pregrado y posgrado?",
            "estrato": "Como varia el bienestar por estrato socioeconomico?",
            "dimensiones": "Cual es el perfil por dimensiones de bienestar?",
            "edad": "Cual es la correlacion entre edad y bienestar?",
            "regresion": "Que factores predicen el bienestar?",
            "validar": "Valida los resultados contra el estudio anterior",
            "validacion": "Haz una validacion completa del pipeline",
        }

        mapped = command_map.get(question.lower(), question)

        # La busqueda web la detecta y ejecuta el orquestador (ask)
        if question.lower().startswith("buscar "):
            print("\nBuscando en bases academicas (Crossref/PubMed)...")

        print(f"\nAnalizando: '{mapped}'...")
        print("-" * 40)

        response = agent.ask(mapped)

        if response.success:
            print(response.interpretation)

            if response.warnings:
                print("\nAdvertencias:")
                for w in response.warnings:
                    print(f"  - {w}")

            # Mostrar trazabilidad (solo el dataset REALMENTE usado)
            t = response.traceability
            print(f"\n[Trazabilidad] Dataset: {t.get('dataset') or 'ninguno'} | "
                  f"Metodo: {t.get('method') or 'N/A'} | "
                  f"Registros: {t.get('n_records') or 'N/A'}")
        else:
            print(f"Error: {response.error}")
            print(response.interpretation)

        print()


if __name__ == "__main__":
    main()

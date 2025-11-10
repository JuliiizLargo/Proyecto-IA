# src/app.py
import os
import argparse
from src.agentes.agente_extraccion import AgenteExtraccion
from src.agentes.agente_analisis import AgenteAnalisis
from src.agentes.agente_respuesta import AgenteRespuesta

def construir_indice(data_dir: str, tam_chunk: int = 300, modelo: str = "all-MiniLM-L6-v2"):
    extractor = AgenteExtraccion(data_dir)
    print("[*] Extrayendo y creando chunks...")
    chunks_meta = extractor.procesar(tam_chunk)
    print(f"[*] Chunks creados: {len(chunks_meta)}")

    analisis = AgenteAnalisis(modelo_name=modelo)
    print("[*] Generando embeddings e indexando...")
    store = analisis.indexar_chunks(chunks_meta)
    print("[*] Index creado.")
    return analisis

def modo_interactivo(analisis: AgenteAnalisis, api_key: str):
    respuesta_agent = AgenteRespuesta(api_key)
    print("\nModo interactivo. Escribe 'salir' para terminar.\n")
    while True:
        pregunta = input("Pregunta > ").strip()
        if pregunta.lower() in ["salir", "exit", "quit"]:
            break
        fragmentos = analisis.buscar_similares(pregunta, top_k=4)
        if not fragmentos:
            print("No hay fragmentos indexados.")
            continue
        out = respuesta_agent.generar_respuesta(pregunta, fragmentos)
        print("\n--- RESPUESTA ---")
        print(out)
        print("-----------------\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", default="data/apuntes/", help="Carpeta con apuntes")
    parser.add_argument("--chunk", type=int, default=300)
    parser.add_argument("--modelo", default="all-MiniLM-L6-v2")
    parser.add_argument("--api_key", default=None, help="OPENAI API KEY (si no está en env)")
    args = parser.parse_args()

    analisis = construir_indice(args.data, tam_chunk=args.chunk, modelo=args.modelo)
    modo_interactivo(analisis, args.api_key)

if __name__ == "__main__":
    main()

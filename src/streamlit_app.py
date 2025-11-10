# src/streamlit_app.py
import os
import streamlit as st
from pathlib import Path
from src.agentes.agente_extraccion import AgenteExtraccion
from src.agentes.agente_analisis import AgenteAnalisis
from src.agentes.agente_respuesta import AgenteRespuesta

# Configuración de la página
st.set_page_config(page_title="🧠 Asistente de Apuntes", layout="wide")
st.title("🧠 Asistente de Búsqueda de Apuntes")

# Sidebar para configuración
with st.sidebar:
    st.header("Configuración")
    
    # Configuración de la API Key
    api_key = st.text_input(
        "🔑 Google API Key",
        type="password",
        help="Obtén tu API key de Google AI Studio: https://aistudio.google.com/"
    )
    
    # Configuración de la carpeta de datos
    data_dir = st.text_input("📂 Carpeta con apuntes", value="data/apuntes/")
    
    # Configuración de los chunks
    tam_chunk = st.slider(
        "📏 Tamaño de los chunks (palabras)",
        min_value=100,
        max_value=1000,
        value=300,
        step=50,
        help="Tamaño de los fragmentos en que se dividirán los documentos"
    )
    
    # Botón para indexar
    if st.button("🔄 Indexar apuntes", use_container_width=True):
        if not api_key:
            st.error("❌ Por favor, ingresa tu Google API Key")
            st.stop()
            
        if not os.path.exists(data_dir):
            st.error(f"❌ La carpeta {data_dir} no existe")
            st.stop()
            
        with st.spinner("🔍 Extrayendo y creando índice..."):
            try:
                extractor = AgenteExtraccion(data_dir)
                chunks_meta = extractor.procesar(tam_chunk)
                
                if not chunks_meta:
                    st.warning("⚠️ No se encontraron documentos para indexar")
                    st.stop()
                
                with st.expander("📊 Estadísticas de indexación", expanded=False):
                    st.write(f"📄 Documentos procesados: {len(set(c['documento'] for c in chunks_meta))}")
                    st.write(f"🧩 Chunks creados: {len(chunks_meta)}")
                    
                    # Mostrar documentos indexados
                    docs = sorted(set(c['documento'] for c in chunks_meta))
                    st.write("📋 Documentos indexados:")
                    for doc in docs:
                        st.write(f"  - {doc}")
                
                analisis = AgenteAnalisis()
                analisis.indexar_chunks(chunks_meta)
                st.session_state["analisis"] = analisis
                st.session_state["chunks_meta"] = chunks_meta
                st.success("✅ Indexado completado")
                
            except Exception as e:
                st.error(f"❌ Error al indexar: {str(e)}")
                st.exception(e)  # Muestra el traceback completo

# Área principal de búsqueda
st.header("🔍 Buscar en los apuntes")

# Mostrar estado actual
if "analisis" not in st.session_state:
    st.warning("⚠️ Por favor, indexa los apuntes primero usando el panel lateral")
    st.stop()

# Campo de búsqueda
pregunta = st.text_input(
    "Escribe tu pregunta:",
    placeholder="Ej: ¿Qué son las redes neuronales?",
    help="Escribe tu pregunta sobre el contenido de los apuntes"
)

# Botón de búsqueda
if st.button("🔍 Buscar", type="primary") and pregunta:
    with st.spinner("🤔 Procesando tu pregunta..."):
        try:
            analisis = st.session_state.get("analisis")
            fragmentos = analisis.buscar_similares(pregunta, top_k=4)
            
            if not fragmentos:
                st.warning("No se encontraron fragmentos relevantes")
                st.stop()
                
            # Mostrar respuesta
            with st.container():
                st.subheader("💡 Respuesta")
                with st.spinner("Generando respuesta..."):
                    sr = AgenteRespuesta(api_key)
                    respuesta = sr.generar_respuesta(pregunta, fragmentos)
                    st.markdown(f"{respuesta}")
            
            # Mostrar fragmentos
            with st.expander("📚 Ver fragmentos usados", expanded=False):
                for f in fragmentos:
                    st.markdown(f"### 📄 {f['documento']} (Chunk {f['chunk_id'] + 1})")
                    st.markdown(f"```\n{f['texto']}\n```")
                    st.write("---")
                    
        except Exception as e:
            st.error(f"❌ Error al buscar: {str(e)}")

# Footer
st.markdown("---")
st.caption("🧠 Asistente de Apuntes - Procesamiento de lenguaje natural con Gemini")

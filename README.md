# 🧠 Asistente de Búsqueda de Apuntes  
**Proyecto Final – Introducción a la Inteligencia Artificial**

Este proyecto implementa un asistente que permite **buscar información dentro de apuntes o documentos (PDF, TXT, imágenes)** utilizando técnicas básicas de **Inteligencia Artificial**:

- Extracción directa y OCR con **Tesseract**
- Segmentación (chunks)
- Generación de **embeddings**
- Búsqueda por **similitud semántica**
- **Base de datos vectorial** con FAISS (o búsqueda por coseno si FAISS no está disponible)
- Arquitectura **multiagente en LangChain**
- Interfaz **CLI y Streamlit**
- Generación de respuestas con **Gemini (Google Generative AI)**

---

## 🖥️ Requisitos del sistema (Windows)

### 🔹 1. Instalar **Python 3.9 o superior**
Descargar desde 👉 [https://www.python.org/downloads/windows/](https://www.python.org/downloads/windows/)

Durante la instalación:
- ✅ Marca la opción **“Add Python to PATH”**
- Luego haz clic en **Customize Installation → Next → Install**

Verifica que quedó bien instalado:
```bash
python --version

# 🧠 Asistente de Búsqueda de Apuntes  
**Proyecto Final – Introducción a la Inteligencia Artificial**

Este proyecto implementa un asistente que permite **buscar información dentro de apuntes o documentos (PDF, TXT, imágenes)** utilizando técnicas de **Inteligencia Artificial**. El sistema está compuesto por tres agentes principales que trabajan en conjunto para proporcionar respuestas precisas basadas en los documentos proporcionados.

## 🚀 Características principales

- Extracción de texto de múltiples formatos: **PDF, TXT, PNG, JPG/JPEG**
- **OCR integrado** con Tesseract para procesamiento de imágenes
- **Segmentación inteligente** de documentos en chunks manejables
- Generación de **embeddings** con modelos pre-entrenados
- Búsqueda por **similitud semántica**
- **Base de datos vectorial** para búsquedas eficientes
- Generación de respuestas con **Gemini (Google Generative AI)**
- Interfaz **CLI y Streamlit**

## 🏗️ Arquitectura del Sistema

### Agentes Principales

1. **Agente de Extracción**
   - Extrae texto de diferentes formatos de archivo
   - Soporta PDF, TXT e imágenes (con OCR)
   - Divide el contenido en fragmentos manejables

2. **Agente de Análisis**
   - Genera embeddings del texto
   - Almacena y gestiona los vectores de los documentos
   - Realiza búsquedas semánticas

3. **Agente de Respuesta**
   - Genera respuestas utilizando Google's Gemini
   - Combina la pregunta con el contexto relevante
   - Proporciona respuestas precisas y contextuales

## �️ Instalación

### Requisitos del sistema

- Python 3.8 o superior
- Tesseract OCR instalado en el sistema
- API Key de Google AI (para Gemini)

### Pasos de instalación

1. **Clonar el repositorio**
   ```bash
   git clone https://github.com/tu-usuario/proyecto-ia-apuntes.git
   cd proyecto-ia-apuntes
   ```

2. **Crear y activar un entorno virtual (recomendado)**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # En Windows
   ```

3. **Instalar dependencias**
   ```bash
   pip install -r requirements.txt
   ```

4. **Instalar Tesseract OCR**
   - Descargar e instalar desde: https://github.com/UB-Mannheim/tesseract/wiki
   - Asegurarse de agregar Tesseract al PATH del sistema

5. **Configurar variables de entorno**
   - Copiar el archivo `.env.example` a `.env`
   - Agregar tu API Key de Google AI:
     ```
     GOOGLE_API_KEY=tu_api_key_aquí
     ```

## 🚀 Uso

### Interfaz de Línea de Comandos (CLI)

```bash
# Procesar documentos en la carpeta data/
python main.py --process

# Realizar una consulta
python main.py --query "Tu pregunta aquí"
```

### Interfaz Web con Streamlit

```bash
streamlit run app.py
```

## 📁 Estructura del Proyecto

```
proyecto-ia-apuntes/
├── data/                   # Carpeta para documentos de entrada
├── src/
│   ├── agentes/           # Agentes del sistema
│   │   ├── agente_extraccion.py
│   │   ├── agente_analisis.py
│   │   └── agente_respuesta.py
│   └── core/              # Módulos principales
│       ├── chunking.py
│       ├── embeddings.py
│       └── vector_store.py
├── .env.example          # Plantilla de variables de entorno
├── requirements.txt      # Dependencias
├── main.py               # Punto de entrada CLI
└── app.py               # Aplicación Streamlit
```

## 📝 Dependencias Principales

- **PyPDF2**: Procesamiento de archivos PDF
- **Pillow & Pytesseract**: Procesamiento de imágenes y OCR
- **sentence-transformers**: Generación de embeddings
- **google-generativeai**: Integración con Google's Gemini
- **streamlit**: Interfaz web interactiva
- **python-dotenv**: Manejo de variables de entorno

## 🤝 Contribución

Las contribuciones son bienvenidas. Por favor, abre un issue primero para discutir los cambios propuestos.

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

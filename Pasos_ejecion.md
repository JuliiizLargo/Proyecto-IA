# Crear entorno virtual
python -m venv venv

# Activar entorno (Windows PowerShell)
.\venv\Scripts\Activate.ps1
# Si aparece error de ejecución:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
# Luego vuelve a activar
.\venv\Scripts\Activate.ps1

# Actualizar pip
python -m pip install --upgrade pip

# Instalar dependencias
pip install -r requirements.txt

copy .env.example .env

# Reemplaza TU_API_KEY con tu clave real
$env:GOOGLE_API_KEY = "TU_API_KEY"

python run_app.py

_______________________--

Error: Tesseract no encontrado
Descargar e instalar Tesseract para Windows: Enlace de descarga
Añadir al PATH o configurar la ruta en el código
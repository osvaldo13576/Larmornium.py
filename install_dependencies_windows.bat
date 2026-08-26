@echo off
REM Instala las dependencias de Larmornium dentro de un entorno virtual (venv).
setlocal

set "PROJECT_ROOT=%~dp0"
set "VENV_DIR=%PROJECT_ROOT%venv"

where python >nul 2>nul
if errorlevel 1 (
    echo Error: no se encontro Python en el PATH.
    exit /b 1
)

if not exist "%VENV_DIR%" (
    python -m venv "%VENV_DIR%"
)

set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"

"%VENV_PYTHON%" -m pip install --upgrade pip
"%VENV_PYTHON%" -m pip install -r "%PROJECT_ROOT%requirements.txt"

echo.
echo Entorno virtual listo en: %VENV_DIR%
echo Ejecuta: venv\Scripts\activate.bat

endlocal

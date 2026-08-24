#!/usr/bin/env bash
# Dependencias para Larmornium (venv).

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_ROOT/venv"

PYTHON_BIN=""
for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
        PYTHON_BIN="$candidate"
        break
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "Error: no se encontró un intérprete de Python (python3/python)." >&2
    exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

VENV_PYTHON="$VENV_DIR/bin/python"

"$VENV_PYTHON" -m pip install --upgrade pip
"$VENV_PYTHON" -m pip install pydicom numpy PySide6 tifffile

echo ""
echo "Entorno virtual listo en: $VENV_DIR"
echo "Activalo con: source venv/bin/activate"


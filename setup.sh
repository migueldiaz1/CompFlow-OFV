#!/usr/bin/env bash
# setup.sh (Linux/macOS -- see setup.bat for Windows)
# =========
# Creates the virtual environment this notebook runs in (venv/) and
# registers it as the standard "Python 3" Jupyter kernel. venv/ is not
# committed to git -- run this once after cloning, then open
# compflow_demo.ipynb and pick the "Python 3" kernel.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# --system-site-packages: reuse an already-installed torch/numpy/opencv/scipy
# instead of re-downloading them (useful if you already have a CUDA torch
# set up system-wide). Drop the flag for a fully isolated environment.
python3 -m venv --system-site-packages venv
venv/bin/pip install --upgrade pip
venv/bin/pip install jupyter ipykernel

# Make sure CompFlow's core dependencies are actually importable (the
# --system-site-packages ones might not exist on your system) -- install
# whatever's missing from the repo's own requirements.txt.
venv/bin/python - <<'PY'
import importlib, subprocess, sys
missing = [m for m in ("torch", "numpy", "cv2", "scipy") if importlib.util.find_spec(m) is None]
if missing:
    print(f"installing missing core deps: {missing}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
PY

venv/bin/python -m ipykernel install --user --name=python3 --display-name "Python 3"

echo
echo "Done. Open compflow_demo.ipynb and select the 'Python 3' kernel"
echo "(jupyter notebook / jupyter lab, or your IDE's kernel picker)."

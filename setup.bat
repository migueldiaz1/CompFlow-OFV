@echo off
rem setup.bat (Windows -- see setup.sh for Linux/macOS)
rem ======================================================
rem Creates the virtual environment this notebook runs in (venv\) and
rem registers it as the standard "Python 3" Jupyter kernel. venv\ is not
rem committed to git -- run this once after cloning, then open
rem compflow_demo.ipynb and pick the "Python 3" kernel.
setlocal
cd /d "%~dp0"

python -m venv --system-site-packages venv
if errorlevel 1 goto :error

venv\Scripts\python.exe -m pip install --upgrade pip
venv\Scripts\python.exe -m pip install jupyter ipykernel

rem Make sure CompFlow's core dependencies are actually importable (the
rem --system-site-packages ones might not exist on your system) -- install
rem whatever's missing from the repo's own requirements.txt.
venv\Scripts\python.exe -c "import importlib, subprocess, sys; missing=[m for m in ('torch','numpy','cv2','scipy') if importlib.util.find_spec(m) is None]; subprocess.check_call([sys.executable,'-m','pip','install','-r','requirements.txt']) if missing else None"
if errorlevel 1 goto :error

venv\Scripts\python.exe -m ipykernel install --user --name=python3 --display-name "Python 3"

echo.
echo Done. Open compflow_demo.ipynb and select the 'Python 3' kernel
echo (jupyter notebook / jupyter lab, or your IDE's kernel picker).
goto :eof

:error
echo Setup failed -- see the error above.
exit /b 1

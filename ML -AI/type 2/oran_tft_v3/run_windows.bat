@echo off
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (set PY=py -3.11) else (set PY=python)
%PY% --version
if errorlevel 1 (echo Install Python 3.11 from https://www.python.org/downloads/ and select Add Python to PATH. & pause & exit /b 1)
if not exist .venv\Scripts\python.exe %PY% -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (echo Package installation failed. Check internet and Python version. & pause & exit /b 1)
if not exist data\oran_kpis.csv python src\generate_data.py
if errorlevel 1 (pause & exit /b 1)
if not exist models\compact_tft.pt python src\train.py
if errorlevel 1 (pause & exit /b 1)
streamlit run app.py
pause

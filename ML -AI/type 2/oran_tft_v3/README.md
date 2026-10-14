# O-RAN Multi-KPI Forecasting — VS Code project

This is a laptop-only student prototype for the project **Temporal Fusion Transformer-Based Multi-KPI Forecasting for Proactive Open RAN Control**. It creates simulated KPI data, trains a compact TFT-style sequence model with LSTM + attention and P10/P50/P90 outputs, predicts 12 future 15-minute steps, and displays a Streamlit dashboard with threshold-based alerts.

**Important accuracy note:** this is an educational TFT-style implementation, not the full canonical Temporal Fusion Transformer library model. It uses simulated data and does not connect to a live O-RAN network or actually change radio resources. Real O-RAN integration and validation remain future work.

## Easiest run (Windows)
1. Install Python 3.11 (64-bit) from https://www.python.org/downloads/ and enable **Add Python to PATH**.
2. Extract the ZIP.
3. Open the extracted `oran_tft_v3` folder in VS Code.
4. Double-click `run_windows.bat`. It creates a virtual environment, installs packages, generates data, trains the model and starts the dashboard.
5. In the terminal, open the local URL printed by Streamlit, usually `http://localhost:8501`.

## Manual run in VS Code terminal (PowerShell)
Run each command from the project root folder:
```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python src\generate_data.py
python src\train.py
streamlit run app.py
```
If PowerShell blocks activation, use Command Prompt terminal and run `.venv\Scripts\activate.bat`, or use the `run_windows.bat` file.

## What the output means
- Current KPI cards: latest simulated values.
- Forecast graph: P50 central prediction and P10–P90 uncertainty band.
- Alert: compares the maximum forecast P90 PRB utilization to 90% and latency to 40 ms (moderate alert begins at PRB 75% or latency 30 ms).
- The 48-step input is 12 hours; 12 forecast steps are 3 hours because each step is 15 minutes.

## Troubleshooting
- `python is not recognized`: reinstall Python 3.11 and check Add Python to PATH, then restart VS Code.
- `No module named ...`: activate `.venv` and run `pip install -r requirements.txt`.
- First install/training may take several minutes. The model trains on CPU and does not need a GPU.

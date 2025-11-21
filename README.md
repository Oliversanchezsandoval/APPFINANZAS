# App de Streamlit – Valuación & Portafolio

App educativa para consultar acciones (yfinance) con gráficas Plotly y construir un portafolio con métricas clave.

## Características
- Sidebar con navegación entre módulos.
- Consulta de acciones: velas (Plotly), 10 highlights financieros, comparación vs S&P 500.
- Portafolio/Simulador: comparación multi-activo, pesos personalizados, métricas (retorno/vol/Sharpe/Beta), y simulador por compras.

## Requisitos
- Python 3.10+
- Paquetes en requirements.txt

## Ejecución local (macOS)
cd ~/Desktop/streamlit-valuacion
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py

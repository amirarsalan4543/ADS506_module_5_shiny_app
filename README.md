# Australian Wine Sales Forecasting

Shiny for Python app comparing TSLM, auto-ETS, and auto-ARIMA models on `AustralianWines.csv` (1980–1994).

## Features
- Interactive varietal, date range, train/validation split, and forecast horizon controls
- Historical overview plot and forecast comparison plot with 95% prediction intervals
- Model specifications (`ETS` and `ARIMA`) and Train vs. Validation accuracy (`RMSE`, `MAE`, `MAPE`)
- CSV download button for model forecasts

## Run Locally
```bash
pip install -r requirements.txt
shiny run app.py
```

*Note: LLM assistance was used to draft boilerplate code.*

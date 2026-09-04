# GRIDPULSE — Electricity Demand Prediction Backend

This package contains the trained XGBoost model, complete hourly dataset, feature engineering, forecasting pipeline, and FastAPI backend.

## Run

1. Install dependencies:
pip install -r requirements.txt

2. Start the API:
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

3. Test:
http://127.0.0.1:8000/api/health

## Frontend API

Same computer: http://127.0.0.1:8000/api

Different computer on the same network: http://BACKEND_IP:8000/api

Example: http://172.25.181.39:8000/api

## Endpoints

GET /api/health
GET /api/dashboard
GET /api/forecast
GET /api/forecast/7d
GET /api/risk
GET /api/weather
GET /api/insights
GET /api/metrics
POST /api/what-if

## Model and data

Model: XGBoost regression
Dataset: 4,416 continuous hourly observations from 2026-03-01 through 2026-08-31.
Dataset validation: 0 duplicate timestamps, 0 non-hourly intervals, 0 missing values.

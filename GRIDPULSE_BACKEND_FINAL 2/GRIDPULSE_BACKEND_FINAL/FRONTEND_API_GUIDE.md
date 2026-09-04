# GRIDPULSE — Frontend API Integration Guide

## API BASE URL

Same computer:
http://127.0.0.1:8000/api

Different computer on the same network:
http://BACKEND_IP:8000/api

Example:
http://172.25.181.39:8000/api

## 24-HOUR FORECAST

GET /api/forecast

Use these fields for the forecast chart:
- time
- predicted
- lower
- upper
- actual
- temperature
- humidity

For future forecast hours, actual is null. Do not invent future actual values.

## 7-DAY FORECAST

GET /api/forecast/7d

Useful fields:
- date
- peak
- average
- minimum

## GRID RISK

GET /api/risk

Useful fields:
- level
- utilization
- peakMw
- peakTime
- marginMw
- factors

## WEATHER

GET /api/weather

Useful fields:
- region
- temperature
- feelsLike
- humidity
- condition
- solarGeneration
- wind
- cloudCover
- solarRadiation
- source

## INSIGHTS

GET /api/insights

Use the returned insight data rather than hardcoding insight values.

## DASHBOARD

GET /api/dashboard

This combines forecast, risk, weather and insights information.

## MODEL METRICS

GET /api/metrics

Use this endpoint for the model-performance section.

## WHAT-IF LAB

POST /api/what-if

Example JSON body:

{ "temperature": 35, "humidity": 60, "solarGeneration": 300, "demandGrowth": 5 }

Response includes:
- baselinePeak
- simulatedPeak
- change
- utilization
- risk
- forecast

## MOCK DATA

Set USE_MOCK_DATA to false when using the backend.
The frontend should not overwrite API results with hardcoded mock values.

## NETWORK SETUP

Start the backend with:
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

Then use the backend computers local IP in the frontend API base URL.

Verify connectivity from the frontend computer using:
http://BACKEND_IP:8000/api/health

The response should contain status: ok and modelLoaded: true.

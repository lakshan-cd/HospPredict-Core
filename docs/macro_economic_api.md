# Macro-Economic API Documentation

## Overview

The Macro-Economic API provides comprehensive access to the macro-economic analysis module for Sri Lankan hospitality sector stock price forecasting. This API enables users to access trained models, make predictions, view analysis results, and process new company data.

## Base URL

```
http://localhost:8000/api/v1/macro-economic
```

## Available Companies

The API currently supports the following companies with trained models:

- AITKEN_SPENCE_HOTEL_HOLDINGS_PLC
- ASIAN_HOTELS_AND_PROPERTIES_PLC
- BROWNS_BEACH_HOTELS_PLC
- CEYLON_HOTELS_CORPORATION_PLC
- DOLPHIN_HOTELS_PLC
- EDEN_HOTEL_LANKA_PLC
- GALADARI_HOTELS_LANKA_PLC
- HOTEL_SIGIRIYA_PLC
- JOHN_KEELLS_HOTELS_PLC
- MAHAWELI_REACH_HOTELS_PLC
- PEGASUS_HOTELS_OF_CEYLON_PLC
- RENUKA_CITY_HOTELS_PLC
- RENUKA_HOTELS_PLC
- SIGIRIYA_VILLAGE_HOTELS_PLC
- TAL_LANKA_HOTELS_PLC
- TANGERINE_BEACH_HOTELS_PLC
- THE_LIGHTHOUSE_HOTEL_PLC
- THE_NUWARA_ELIYA_HOTELS_COMPANY_PLC

## API Endpoints

### 1. Get Available Companies

**Endpoint:** `GET /companies`

**Description:** Retrieve list of all available companies with trained models.

**Response:**
```json
{
  "companies": [
    "AITKEN_SPENCE_HOTEL_HOLDINGS_PLC",
    "ASIAN_HOTELS_AND_PROPERTIES_PLC",
    ...
  ],
  "total_count": 19
}
```

### 2. Get Company Information

**Endpoint:** `GET /companies/{company_name}`

**Description:** Get detailed information about a specific company's data.

**Parameters:**
- `company_name` (string): Name of the company (use underscores instead of spaces)

**Response:**
```json
{
  "company_name": "AITKEN_SPENCE_HOTEL_HOLDINGS_PLC",
  "data_points": 2495,
  "date_range": {
    "start": "2014-01-02",
    "end": "2024-01-15"
  },
  "features": {
    "stock_features": ["Open (Rs.)", "High (Rs.)", "Low (Rs.)", "Close (Rs.)", "TradeVolume"],
    "macro_features": ["Tourism Arrivals", "Money Supply", "Exchange Rate", "Inflation rate"]
  },
  "sequences": {
    "X_stock_shape": [2432, 30, 1],
    "X_macro_shape": [2432, 30, 4],
    "y_shape": [2432]
  }
}
```

### 3. Make Stock Price Prediction

**Endpoint:** `POST /predict`

**Description:** Predict stock price for a specific company.

**Request Body:**
```json
{
  "company_name": "AITKEN_SPENCE_HOTEL_HOLDINGS_PLC",
  "days_ahead": 1
}
```

**Response:**
```json
{
  "company_name": "AITKEN_SPENCE_HOTEL_HOLDINGS_PLC",
  "current_price": 125.50,
  "predicted_price": 127.80,
  "price_change": 2.30,
  "price_change_percent": 1.83,
  "prediction_date": "2024-01-15",
  "model_confidence": "high"
}
```

### 4. Get Model Performance

**Endpoint:** `GET /performance/{company_name}`

**Description:** Get model performance metrics for a company.

**Response:**
```json
{
  "company_name": "AITKEN_SPENCE_HOTEL_HOLDINGS_PLC",
  "metrics": {
    "rmse": 1.9006,
    "mae": 1.3587,
    "mape": 2.15
  },
  "data_points": 2432,
  "date_range": {
    "start": 0,
    "end": 2431
  }
}
```

### 5. Get Feature Importance

**Endpoint:** `GET /feature-importance/{company_name}`

**Description:** Get SHAP feature importance analysis for a company.

**Response:**
```json
{
  "company_name": "AITKEN_SPENCE_HOTEL_HOLDINGS_PLC",
  "feature_importance": {
    "summary_plot": "/api/v1/macro-economic/plots/shap/AITKEN_SPENCE_HOTEL_HOLDINGS_PLC/summary",
    "bar_plot": "/api/v1/macro-economic/plots/shap/AITKEN_SPENCE_HOTEL_HOLDINGS_PLC/bar"
  },
  "plots_available": true
}
```

### 6. Get Granger Causality Analysis

**Endpoint:** `GET /granger-causality/{company_name}`

**Description:** Get Granger causality analysis results for a company.

**Response:**
```json
{
  "company_name": "AITKEN_SPENCE_HOTEL_HOLDINGS_PLC",
  "feature_pvalues": {
    "Tourism Arrivals": 0.0234,
    "Money Supply": 0.0456,
    "Exchange Rate": 0.0123,
    "Inflation rate": 0.0789
  },
  "significant_features": ["Tourism Arrivals", "Money Supply", "Exchange Rate"],
  "heatmap_available": true,
  "heatmap_url": "/api/v1/macro-economic/plots/granger/AITKEN_SPENCE_HOTEL_HOLDINGS_PLC/heatmap"
}
```

### 7. Process New Company Data

**Endpoint:** `POST /new-company`

**Description:** Process new company data and make predictions using existing models.

**Request Body:**
```json
{
  "company_name": "NEW_HOTEL_COMPANY_PLC",
  "stock_data": [
    {
      "Trade Date": "1/2/14",
      "Open (Rs.)": 69.9,
      "High (Rs.)": 73.0,
      "Low (Rs.)": 69.9,
      "Close (Rs.)": 71.9,
      "TradeVolume": 6
    }
  ],
  "macro_data": [
    {
      "Date": "2014-01-02",
      "Tourism Arrivals": 125000,
      "Money Supply": 2500000,
      "Exchange Rate": 130.5,
      "Inflation rate": 4.2
    }
  ]
}
```

**Response:**
```json
{
  "company_name": "NEW_HOTEL_COMPANY_PLC",
  "processed": true,
  "data_points": 1200,
  "current_price": 125.50,
  "predicted_price": 127.80,
  "price_change": 2.30,
  "price_change_percent": 1.83,
  "model_used": "AITKEN_SPENCE_HOTEL_HOLDINGS_PLC",
  "confidence": "medium"
}
```

### 8. Get Prediction Plot

**Endpoint:** `GET /plots/prediction/{company_name}`

**Description:** Get prediction plot image for a company.

**Response:** PNG image file

### 9. Get SHAP Summary Plot

**Endpoint:** `GET /plots/shap/{company_name}/summary`

**Description:** Get SHAP summary plot image for a company.

**Response:** PNG image file

### 10. Get SHAP Bar Plot

**Endpoint:** `GET /plots/shap/{company_name}/bar`

**Description:** Get SHAP bar plot image for a company.

**Response:** PNG image file

### 11. Get Granger Causality Heatmap

**Endpoint:** `GET /plots/granger/{company_name}/heatmap`

**Description:** Get Granger causality heatmap image for a company.

**Response:** PNG image file

### 12. Get API Summary

**Endpoint:** `GET /summary`

**Description:** Get summary of all available data and models.

**Response:**
```json
{
  "total_companies": 19,
  "available_companies": [
    "AITKEN_SPENCE_HOTEL_HOLDINGS_PLC",
    "ASIAN_HOTELS_AND_PROPERTIES_PLC",
    ...
  ],
  "features": [
    "Tourism Arrivals",
    "Money Supply", 
    "Exchange Rate",
    "Inflation rate"
  ],
  "window_size": 30,
  "data_sources": {
    "processed_data": "data/macro-economic/data/processed",
    "sequences": "data/macro-economic/data/processed/sequences",
    "models": "models/macro-economic/models",
    "outputs": "outputs"
  }
}
```

## Data Formats

### Stock Data Format

For new company data, stock data should include:

```json
{
  "Trade Date": "MM/DD/YY",
  "Open (Rs.)": 69.9,
  "High (Rs.)": 73.0,
  "Low (Rs.)": 69.9,
  "Close (Rs.)": 71.9,
  "TradeVolume": 6
}
```

**Required Fields:**
- `Trade Date`: Date in MM/DD/YY format
- `Close (Rs.)`: Closing stock price in Sri Lankan Rupees

**Optional Fields:**
- `Open (Rs.)`: Opening price
- `High (Rs.)`: Highest price of the day
- `Low (Rs.)`: Lowest price of the day
- `TradeVolume`: Trading volume

### Macro Data Format

For new company data, macro data should include:

```json
{
  "Date": "YYYY-MM-DD",
  "Tourism Arrivals": 125000,
  "Money Supply": 2500000,
  "Exchange Rate": 130.5,
  "Inflation rate": 4.2
}
```

**Required Fields:**
- `Date`: Date in YYYY-MM-DD format
- `Tourism Arrivals`: Number of tourist arrivals
- `Money Supply`: Money supply in millions
- `Exchange Rate`: LKR/USD exchange rate
- `Inflation rate`: Inflation rate percentage

## Error Responses

The API returns standard HTTP error codes:

- `400 Bad Request`: Invalid request data or missing required fields
- `404 Not Found`: Company or data not found
- `500 Internal Server Error`: Server-side processing error

Error response format:
```json
{
  "detail": "Error message describing the issue"
}
```

## Usage Examples

### Python Example

```python
import requests
import json

# Base URL
base_url = "http://localhost:8000/api/v1/macro-economic"

# Get available companies
response = requests.get(f"{base_url}/companies")
companies = response.json()["companies"]

# Make prediction for a company
prediction_data = {
    "company_name": "AITKEN_SPENCE_HOTEL_HOLDINGS_PLC",
    "days_ahead": 1
}
response = requests.post(f"{base_url}/predict", json=prediction_data)
prediction = response.json()

print(f"Predicted price: {prediction['predicted_price']}")
print(f"Price change: {prediction['price_change_percent']}%")
```

### JavaScript Example

```javascript
// Get available companies
fetch('http://localhost:8000/api/v1/macro-economic/companies')
  .then(response => response.json())
  .then(data => {
    console.log('Available companies:', data.companies);
  });

// Make prediction
const predictionData = {
  company_name: 'AITKEN_SPENCE_HOTEL_HOLDINGS_PLC',
  days_ahead: 1
};

fetch('http://localhost:8000/api/v1/macro-economic/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify(predictionData)
})
.then(response => response.json())
.then(data => {
  console.log('Prediction:', data);
});
```

### cURL Examples

```bash
# Get available companies
curl -X GET "http://localhost:8000/api/v1/macro-economic/companies"

# Get company information
curl -X GET "http://localhost:8000/api/v1/macro-economic/companies/AITKEN_SPENCE_HOTEL_HOLDINGS_PLC"

# Make prediction
curl -X POST "http://localhost:8000/api/v1/macro-economic/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "company_name": "AITKEN_SPENCE_HOTEL_HOLDINGS_PLC",
    "days_ahead": 1
  }'

# Get prediction plot
curl -X GET "http://localhost:8000/api/v1/macro-economic/plots/prediction/AITKEN_SPENCE_HOTEL_HOLDINGS_PLC" \
  --output prediction_plot.png
```

## Model Information

### Architecture

The macro-economic module uses a dual-input LSTM model:

1. **Stock Input Branch**: Processes 30-day historical stock price sequences
2. **Macro Input Branch**: Processes 30-day macroeconomic indicator sequences
3. **Combination Layer**: Concatenates both branches and passes through dense layers
4. **Output**: Single value prediction for next day's stock price

### Features

**Stock Features:**
- Closing stock prices (scaled using MinMaxScaler)

**Macroeconomic Features:**
- Tourism Arrivals
- Money Supply (M2)
- Exchange Rate (LKR/USD)
- Inflation Rate

### Training Details

- **Window Size**: 30 days
- **Train/Test Split**: 80/20 chronological split
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam with learning rate 0.001
- **Early Stopping**: Patience of 5 epochs

### Performance Metrics

Models are evaluated using:
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error

## Notes

1. **Company Names**: Use underscores instead of spaces in company names for API calls
2. **Data Requirements**: Minimum 60 data points required for processing new companies
3. **Date Formats**: Stock data uses MM/DD/YY format, macro data uses YYYY-MM-DD format
4. **Scaling**: All data is automatically scaled using MinMaxScaler
5. **Model Transfer**: New companies use the most similar existing model for predictions
6. **Plots**: Generated plots are saved as PNG files and served via API endpoints

## Support

For technical support or questions about the API, please refer to the project documentation or contact the development team. 
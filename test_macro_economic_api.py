#!/usr/bin/env python3
"""
Test script for Macro-Economic API
This script demonstrates how to use the macro-economic API endpoints
"""

import requests
import json
import time
from typing import Dict, Any

# Configuration
BASE_URL = "http://localhost:8000/api/v1/macro-economic"
TEST_COMPANY = "AITKEN_SPENCE_HOTEL_HOLDINGS_PLC"

def test_api_endpoint(endpoint: str, method: str = "GET", data: Dict = None) -> Dict[str, Any]:
    """Test an API endpoint and return the response"""
    url = f"{BASE_URL}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {"success": False, "error": response.text, "status_code": response.status_code}
    
    except Exception as e:
        return {"success": False, "error": str(e)}

def test_available_companies():
    """Test getting available companies"""
    print("Testing: Get Available Companies")
    result = test_api_endpoint("/companies")
    
    if result["success"]:
        companies = result["data"]["companies"]
        print(f"✓ Found {len(companies)} companies")
        print(f"  Sample companies: {companies[:3]}")
    else:
        print(f"✗ Failed: {result['error']}")
    
    print()

def test_company_info():
    """Test getting company information"""
    print("Testing: Get Company Information")
    result = test_api_endpoint(f"/companies/{TEST_COMPANY}")
    
    if result["success"]:
        data = result["data"]
        print(f"✓ Company: {data['company_name']}")
        print(f"  Data points: {data['data_points']}")
        print(f"  Date range: {data['date_range']['start']} to {data['date_range']['end']}")
        print(f"  Features: {len(data['features']['macro_features'])} macro features")
    else:
        print(f"✗ Failed: {result['error']}")
    
    print()

def test_prediction():
    """Test making a prediction"""
    print("Testing: Make Stock Price Prediction")
    prediction_data = {
        "company_name": TEST_COMPANY,
        "days_ahead": 1
    }
    
    result = test_api_endpoint("/predict", method="POST", data=prediction_data)
    
    if result["success"]:
        data = result["data"]
        print(f"✓ Company: {data['company_name']}")
        print(f"  Current price: Rs. {data['current_price']}")
        print(f"  Predicted price: Rs. {data['predicted_price']}")
        print(f"  Price change: Rs. {data['price_change']} ({data['price_change_percent']}%)")
    else:
        print(f"✗ Failed: {result['error']}")
    
    print()

def test_model_performance():
    """Test getting model performance"""
    print("Testing: Get Model Performance")
    result = test_api_endpoint(f"/performance/{TEST_COMPANY}")
    
    if result["success"]:
        data = result["data"]
        metrics = data["metrics"]
        print(f"✓ Company: {data['company_name']}")
        print(f"  RMSE: {metrics['rmse']}")
        print(f"  MAE: {metrics['mae']}")
        print(f"  MAPE: {metrics['mape']}%")
        print(f"  Data points: {data['data_points']}")
    else:
        print(f"✗ Failed: {result['error']}")
    
    print()

def test_feature_importance():
    """Test getting feature importance"""
    print("Testing: Get Feature Importance")
    result = test_api_endpoint(f"/feature-importance/{TEST_COMPANY}")
    
    if result["success"]:
        data = result["data"]
        print(f"✓ Company: {data['company_name']}")
        print(f"  Plots available: {data['plots_available']}")
        if data['feature_importance']:
            print(f"  Available plots: {list(data['feature_importance'].keys())}")
    else:
        print(f"✗ Failed: {result['error']}")
    
    print()

def test_granger_causality():
    """Test getting Granger causality analysis"""
    print("Testing: Get Granger Causality Analysis")
    result = test_api_endpoint(f"/granger-causality/{TEST_COMPANY}")
    
    if result["success"]:
        data = result["data"]
        print(f"✓ Company: {data['company_name']}")
        print(f"  Significant features: {data['significant_features']}")
        print(f"  Heatmap available: {data['heatmap_available']}")
        
        # Show p-values for each feature
        for feature, pvalue in data['feature_pvalues'].items():
            significance = "✓" if pvalue < 0.05 else "✗"
            print(f"  {significance} {feature}: p={pvalue}")
    else:
        print(f"✗ Failed: {result['error']}")
    
    print()

def test_api_summary():
    """Test getting API summary"""
    print("Testing: Get API Summary")
    result = test_api_endpoint("/summary")
    
    if result["success"]:
        data = result["data"]
        print(f"✓ Total companies: {data['total_companies']}")
        print(f"  Features: {data['features']}")
        print(f"  Window size: {data['window_size']}")
        print(f"  Data sources: {list(data['data_sources'].keys())}")
    else:
        print(f"✗ Failed: {result['error']}")
    
    print()

def test_new_company_processing():
    """Test processing new company data"""
    print("Testing: Process New Company Data")
    
    # Sample data for a new company
    new_company_data = {
        "company_name": "TEST_HOTEL_COMPANY_PLC",
        "stock_data": [
            {
                "Trade Date": "1/2/14",
                "Open (Rs.)": 69.9,
                "High (Rs.)": 73.0,
                "Low (Rs.)": 69.9,
                "Close (Rs.)": 71.9,
                "TradeVolume": 6
            },
            {
                "Trade Date": "1/3/14",
                "Open (Rs.)": 71.4,
                "High (Rs.)": 71.4,
                "Low (Rs.)": 64.7,
                "Close (Rs.)": 68.3,
                "TradeVolume": 15
            }
        ],
        "macro_data": [
            {
                "Date": "2014-01-02",
                "Tourism Arrivals": 125000,
                "Money Supply": 2500000,
                "Exchange Rate": 130.5,
                "Inflation rate": 4.2
            },
            {
                "Date": "2014-01-03",
                "Tourism Arrivals": 126000,
                "Money Supply": 2510000,
                "Exchange Rate": 130.8,
                "Inflation rate": 4.1
            }
        ]
    }
    
    result = test_api_endpoint("/new-company", method="POST", data=new_company_data)
    
    if result["success"]:
        data = result["data"]
        print(f"✓ Company: {data['company_name']}")
        print(f"  Processed: {data['processed']}")
        if data.get('predicted_price'):
            print(f"  Predicted price: Rs. {data['predicted_price']}")
            print(f"  Model used: {data['model_used']}")
    else:
        print(f"✗ Failed: {result['error']}")
    
    print()

def test_plot_endpoints():
    """Test plot endpoints"""
    print("Testing: Plot Endpoints")
    
    plot_endpoints = [
        f"/plots/prediction/{TEST_COMPANY}",
        f"/plots/shap/{TEST_COMPANY}/summary",
        f"/plots/shap/{TEST_COMPANY}/bar",
        f"/plots/granger/{TEST_COMPANY}/heatmap"
    ]
    
    for endpoint in plot_endpoints:
        print(f"  Testing {endpoint}...")
        result = test_api_endpoint(endpoint)
        
        if result["success"]:
            print(f"    ✓ Available")
        else:
            print(f"    ✗ Not available: {result.get('status_code', 'Unknown error')}")

def main():
    """Run all tests"""
    print("=" * 60)
    print("MACRO-ECONOMIC API TEST SUITE")
    print("=" * 60)
    print()
    
    # Test basic endpoints
    test_available_companies()
    test_company_info()
    test_prediction()
    test_model_performance()
    test_feature_importance()
    test_granger_causality()
    test_api_summary()
    
    # Test advanced endpoints
    test_new_company_processing()
    test_plot_endpoints()
    
    print("=" * 60)
    print("TEST SUITE COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    main() 
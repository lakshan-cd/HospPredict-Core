#!/usr/bin/env python3
"""
Example: Processing New Company Data with Macro-Economic API
This script demonstrates how to use the API to process new company data
and make predictions using existing trained models.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# Configuration
API_BASE_URL = "http://localhost:8000/api/v1/macro-economic"

def generate_sample_data(company_name: str, days: int = 100) -> dict:
    """
    Generate sample stock and macro data for a new company
    
    Args:
        company_name: Name of the company
        days: Number of days of data to generate
        
    Returns:
        Dictionary containing stock_data and macro_data
    """
    # Generate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate sample stock data
    base_price = 100.0
    stock_data = []
    
    for i, date in enumerate(date_range):
        # Simulate realistic stock price movements
        price_change = np.random.normal(0, 2)  # Daily price change
        base_price = max(50, base_price + price_change)  # Ensure positive price
        
        stock_data.append({
            "Trade Date": date.strftime("%m/%d/%y"),
            "Open (Rs.)": round(base_price - np.random.uniform(0, 5), 2),
            "High (Rs.)": round(base_price + np.random.uniform(0, 10), 2),
            "Low (Rs.)": round(base_price - np.random.uniform(0, 8), 2),
            "Close (Rs.)": round(base_price, 2),
            "TradeVolume": int(np.random.uniform(1000, 50000))
        })
    
    # Generate sample macro data
    macro_data = []
    base_tourism = 100000
    base_money_supply = 5000000
    base_exchange_rate = 130.0
    base_inflation = 4.0
    
    for i, date in enumerate(date_range):
        # Simulate realistic macro economic movements
        tourism_change = np.random.normal(0, 5000)
        money_supply_change = np.random.normal(0, 100000)
        exchange_rate_change = np.random.normal(0, 2)
        inflation_change = np.random.normal(0, 0.1)
        
        base_tourism = max(50000, base_tourism + tourism_change)
        base_money_supply = max(1000000, base_money_supply + money_supply_change)
        base_exchange_rate = max(100, base_exchange_rate + exchange_rate_change)
        base_inflation = max(1, min(10, base_inflation + inflation_change))
        
        macro_data.append({
            "Date": date.strftime("%Y-%m-%d"),
            "Tourism Arrivals": int(base_tourism),
            "Money Supply": int(base_money_supply),
            "Exchange Rate": round(base_exchange_rate, 2),
            "Inflation rate": round(base_inflation, 2)
        })
    
    return {
        "company_name": company_name,
        "stock_data": stock_data,
        "macro_data": macro_data
    }

def process_new_company(company_data: dict) -> dict:
    """
    Process new company data using the API
    
    Args:
        company_data: Dictionary containing company data
        
    Returns:
        API response
    """
    url = f"{API_BASE_URL}/new-company"
    
    try:
        response = requests.post(url, json=company_data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error processing company data: {e}")
        return None

def get_available_models() -> list:
    """Get list of available trained models"""
    url = f"{API_BASE_URL}/companies"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()["companies"]
    except requests.exceptions.RequestException as e:
        print(f"Error getting available models: {e}")
        return []

def make_prediction(company_name: str, days_ahead: int = 1) -> dict:
    """
    Make prediction for an existing company
    
    Args:
        company_name: Name of the company
        days_ahead: Number of days to predict ahead
        
    Returns:
        Prediction results
    """
    url = f"{API_BASE_URL}/predict"
    data = {
        "company_name": company_name,
        "days_ahead": days_ahead
    }
    
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error making prediction: {e}")
        return None

def get_company_performance(company_name: str) -> dict:
    """
    Get model performance for a company
    
    Args:
        company_name: Name of the company
        
    Returns:
        Performance metrics
    """
    url = f"{API_BASE_URL}/performance/{company_name}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error getting performance: {e}")
        return None

def main():
    """Main example function"""
    print("=" * 60)
    print("MACRO-ECONOMIC API - NEW COMPANY EXAMPLE")
    print("=" * 60)
    print()
    
    # Step 1: Check available models
    print("1. Checking available trained models...")
    available_models = get_available_models()
    if available_models:
        print(f"   ✓ Found {len(available_models)} trained models")
        print(f"   Sample models: {available_models[:3]}")
    else:
        print("   ✗ No trained models found")
        return
    
    print()
    
    # Step 2: Generate sample data for a new company
    print("2. Generating sample data for new company...")
    new_company_name = "EXAMPLE_HOTEL_CORPORATION_PLC"
    sample_data = generate_sample_data(new_company_name, days=120)
    
    print(f"   ✓ Generated data for: {sample_data['company_name']}")
    print(f"   Stock data points: {len(sample_data['stock_data'])}")
    print(f"   Macro data points: {len(sample_data['macro_data'])}")
    print(f"   Date range: {sample_data['stock_data'][0]['Trade Date']} to {sample_data['stock_data'][-1]['Trade Date']}")
    
    print()
    
    # Step 3: Process the new company data
    print("3. Processing new company data...")
    result = process_new_company(sample_data)
    
    if result:
        print(f"   ✓ Successfully processed: {result['company_name']}")
        print(f"   Data points processed: {result['data_points']}")
        
        if result.get('predicted_price'):
            print(f"   Current price: Rs. {result['current_price']}")
            print(f"   Predicted price: Rs. {result['predicted_price']}")
            print(f"   Price change: Rs. {result['price_change']} ({result['price_change_percent']}%)")
            print(f"   Model used: {result['model_used']}")
            print(f"   Confidence: {result['confidence']}")
        else:
            print(f"   Note: {result.get('prediction', 'No prediction available')}")
    else:
        print("   ✗ Failed to process company data")
    
    print()
    
    # Step 4: Make prediction for an existing company
    print("4. Making prediction for existing company...")
    existing_company = available_models[0]  # Use first available model
    prediction = make_prediction(existing_company)
    
    if prediction:
        print(f"   ✓ Prediction for: {prediction['company_name']}")
        print(f"   Current price: Rs. {prediction['current_price']}")
        print(f"   Predicted price: Rs. {prediction['predicted_price']}")
        print(f"   Price change: Rs. {prediction['price_change']} ({prediction['price_change_percent']}%)")
    else:
        print("   ✗ Failed to make prediction")
    
    print()
    
    # Step 5: Get performance metrics
    print("5. Getting model performance...")
    performance = get_company_performance(existing_company)
    
    if performance:
        metrics = performance['metrics']
        print(f"   ✓ Performance for: {performance['company_name']}")
        print(f"   RMSE: {metrics['rmse']}")
        print(f"   MAE: {metrics['mae']}")
        print(f"   MAPE: {metrics['mape']}%")
        print(f"   Data points: {performance['data_points']}")
    else:
        print("   ✗ Failed to get performance metrics")
    
    print()
    
    # Step 6: Show API summary
    print("6. API Summary...")
    try:
        response = requests.get(f"{API_BASE_URL}/summary")
        if response.status_code == 200:
            summary = response.json()
            print(f"   ✓ Total companies: {summary['total_companies']}")
            print(f"   Features: {summary['features']}")
            print(f"   Window size: {summary['window_size']}")
        else:
            print("   ✗ Failed to get API summary")
    except Exception as e:
        print(f"   ✗ Error getting API summary: {e}")
    
    print()
    print("=" * 60)
    print("EXAMPLE COMPLETED")
    print("=" * 60)

def example_with_real_data():
    """
    Example using real data format
    This shows the expected data structure for real company data
    """
    print("\n" + "=" * 60)
    print("REAL DATA FORMAT EXAMPLE")
    print("=" * 60)
    
    # Example of real data structure
    real_data_example = {
        "company_name": "REAL_HOTEL_COMPANY_PLC",
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
    
    print("Required data format:")
    print(json.dumps(real_data_example, indent=2))
    
    print("\nData requirements:")
    print("- Minimum 60 data points for both stock and macro data")
    print("- Stock data: Trade Date (MM/DD/YY), Close (Rs.) required")
    print("- Macro data: Date (YYYY-MM-DD), all 4 macro features required")
    print("- Dates must match between stock and macro data")

if __name__ == "__main__":
    main()
    example_with_real_data() 
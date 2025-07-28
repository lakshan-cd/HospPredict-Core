#!/usr/bin/env python3
"""
Test script to check if the macro-economic API works without numpy serialization errors
"""

import requests
import json

def test_macro_economic_api():
    """Test the macro-economic API endpoints"""
    base_url = "http://localhost:8000/api/v1"
    
    # Test 1: Get available companies
    print("Testing: Get available companies")
    try:
        response = requests.get(f"{base_url}/macro-economic/companies")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Success: Found {data['total_count']} companies")
            if data['companies']:
                company_id = data['companies'][0]['id']
                print(f"  First company: {company_id}")
            else:
                print("  No companies found")
        else:
            print(f"✗ Failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print()
    
    # Test 2: Test new company processing (this was causing the error)
    print("Testing: Process new company data")
    try:
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
        
        response = requests.post(f"{base_url}/macro-economic/new-company", json=new_company_data)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Success: Processed company {data['company_name']}")
            print(f"  Data points: {data['data_points']}")
            if 'predicted_price' in data:
                print(f"  Predicted price: {data['predicted_price']}")
                print(f"  Model used: {data['model_used']}")
            else:
                print(f"  Result: {data.get('prediction', 'Unknown')}")
        else:
            print(f"✗ Failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print()
    print("Test completed!")

if __name__ == "__main__":
    test_macro_economic_api() 
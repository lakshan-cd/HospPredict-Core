#!/usr/bin/env python3
"""
Startup script for Macro-Economic API
This script helps start the API server and provides basic setup instructions
"""

import os
import sys
import subprocess
import time
import requests
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    print("Checking dependencies...")
    
    required_packages = [
        'fastapi',
        'uvicorn', 
        'pandas',
        'numpy',
        'tensorflow',
        'scikit-learn',
        'joblib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Please install missing packages using:")
        print("pip install -r requirements_macro_economic.txt")
        return False
    
    print("  All dependencies are installed!")
    return True

def check_data_files():
    """Check if required data files exist"""
    print("\nChecking data files...")
    
    required_paths = [
        "data/macro-economic/data/processed",
        "data/macro-economic/data/processed/sequences", 
        "models/macro-economic/models",
        "outputs"
    ]
    
    missing_paths = []
    
    for path in required_paths:
        if os.path.exists(path):
            print(f"  ✓ {path}")
        else:
            print(f"  ✗ {path} (missing)")
            missing_paths.append(path)
    
    if missing_paths:
        print(f"\nMissing directories: {', '.join(missing_paths)}")
        print("Please ensure the macro-economic module data is properly set up.")
        return False
    
    # Check for model files
    model_dir = "models/macro-economic/models"
    if os.path.exists(model_dir):
        model_files = [f for f in os.listdir(model_dir) if f.endswith('_lstm_model.keras')]
        if model_files:
            print(f"  ✓ Found {len(model_files)} trained models")
        else:
            print("  ✗ No trained models found")
            return False
    
    return True

def start_server(host="0.0.0.0", port=8000):
    """Start the FastAPI server"""
    print(f"\nStarting Macro-Economic API server...")
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"URL: http://{host}:{port}")
    print("\nPress Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Start the server
        cmd = [
            sys.executable, "-m", "uvicorn", 
            "main:app", 
            "--host", host, 
            "--port", str(port),
            "--reload"
        ]
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error starting server: {e}")

def test_api_connection(host="localhost", port=8000):
    """Test if the API is running"""
    url = f"http://{host}:{port}/api/v1/macro-economic/summary"
    
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ API is running successfully!")
            print(f"  Available companies: {data['total_companies']}")
            return True
        else:
            print(f"✗ API returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Could not connect to API server")
        return False
    except Exception as e:
        print(f"✗ Error testing API: {e}")
        return False

def show_api_info():
    """Show API information and endpoints"""
    print("\n" + "=" * 60)
    print("MACRO-ECONOMIC API INFORMATION")
    print("=" * 60)
    
    print("\nAvailable Endpoints:")
    print("  GET  /api/v1/macro-economic/companies")
    print("  GET  /api/v1/macro-economic/companies/{company_name}")
    print("  POST /api/v1/macro-economic/predict")
    print("  GET  /api/v1/macro-economic/performance/{company_name}")
    print("  GET  /api/v1/macro-economic/feature-importance/{company_name}")
    print("  GET  /api/v1/macro-economic/granger-causality/{company_name}")
    print("  POST /api/v1/macro-economic/new-company")
    print("  GET  /api/v1/macro-economic/plots/prediction/{company_name}")
    print("  GET  /api/v1/macro-economic/plots/shap/{company_name}/summary")
    print("  GET  /api/v1/macro-economic/plots/shap/{company_name}/bar")
    print("  GET  /api/v1/macro-economic/plots/granger/{company_name}/heatmap")
    print("  GET  /api/v1/macro-economic/summary")
    
    print("\nInteractive API Documentation:")
    print("  http://localhost:8000/docs")
    print("  http://localhost:8000/redoc")
    
    print("\nExample Usage:")
    print("  # Get available companies")
    print("  curl http://localhost:8000/api/v1/macro-economic/companies")
    print()
    print("  # Make prediction")
    print("  curl -X POST http://localhost:8000/api/v1/macro-economic/predict \\")
    print("    -H 'Content-Type: application/json' \\")
    print("    -d '{\"company_name\": \"AITKEN_SPENCE_HOTEL_HOLDINGS_PLC\", \"days_ahead\": 1}'")
    
    print("\nTest Scripts:")
    print("  python test_macro_economic_api.py")
    print("  python examples/new_company_example.py")

def main():
    """Main function"""
    print("=" * 60)
    print("MACRO-ECONOMIC API STARTUP")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        print("\nPlease install missing dependencies and try again.")
        return
    
    # Check data files
    if not check_data_files():
        print("\nPlease ensure all required data files are present and try again.")
        return
    
    # Show API information
    show_api_info()
    
    # Ask user what to do
    print("\n" + "=" * 60)
    print("What would you like to do?")
    print("1. Start the API server")
    print("2. Test API connection (if server is already running)")
    print("3. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            start_server()
            break
        elif choice == "2":
            if test_api_connection():
                print("\nAPI is working correctly!")
            else:
                print("\nAPI is not running. Please start the server first.")
            break
        elif choice == "3":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main() 
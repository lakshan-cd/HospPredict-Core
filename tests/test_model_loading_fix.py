#!/usr/bin/env python3
"""
Test script to check if the model loading fix works
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from src.predict_utils import load_rf_model, load_scaler

def test_model_loading():
    """Test if models can be loaded with the new error handling"""
    try:
        print("Testing RF model loading...")
        rf_model = load_rf_model("models/rf_gnn_ensemble.joblib")
        print("✓ RF model loaded successfully")
        
        print("Testing scaler loading...")
        scaler = load_scaler("models/scaler/tabular_scaler.joblib")
        print("✓ Scaler loaded successfully")
        
        return True
    except Exception as e:
        print(f"✗ Error loading models: {e}")
        return False

if __name__ == "__main__":
    success = test_model_loading()
    if success:
        print("\nAll models loaded successfully!")
    else:
        print("\nModel loading failed!")
        sys.exit(1) 
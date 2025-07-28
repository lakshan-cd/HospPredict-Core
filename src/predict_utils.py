import torch
import joblib
import numpy as np
from src.inference_utils import HeteroFTAGNN_v3, compute_period_vol

def load_gnn_model(model_path, in_fm, in_sm, in_tp, in_gf, device='cpu'):
    model = HeteroFTAGNN_v3(in_fm, in_sm, in_tp, in_gf)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def load_rf_model(rf_path):
    try:
        return joblib.load(rf_path)
    except ModuleNotFoundError as e:
        if 'numpy._core' in str(e):
            # Handle numpy version compatibility issue
            print(f"Warning: numpy version compatibility issue detected: {e}")
            print("Attempting to load model with numpy version compatibility fix...")
            
            # Try loading with allow_pickle=True for older numpy versions
            try:
                return joblib.load(rf_path, allow_pickle=True)
            except Exception as e2:
                # If that fails, try with pickle protocol compatibility
                try:
                    import pickle
                    with open(rf_path, 'rb') as f:
                        return pickle.load(f, encoding='latin1')
                except Exception as e3:
                    raise Exception(f"Failed to load RF model due to numpy version incompatibility. "
                                  f"Original error: {e}. Secondary error: {e2}. "
                                  f"Pickle error: {e3}. "
                                  f"Please consider upgrading numpy or re-saving the model.")
        else:
            raise e

def load_scaler(scaler_path):
    try:
        return joblib.load(scaler_path)
    except ModuleNotFoundError as e:
        if 'numpy._core' in str(e):
            # Handle numpy version compatibility issue for scaler too
            print(f"Warning: numpy version compatibility issue detected for scaler: {e}")
            try:
                return joblib.load(scaler_path, allow_pickle=True)
            except:
                try:
                    return joblib.load(scaler_path, encoding='latin1')
                except Exception as e2:
                    raise Exception(f"Failed to load scaler due to numpy version incompatibility. "
                                  f"Original error: {e}. Secondary error: {e2}.")
        else:
            raise e
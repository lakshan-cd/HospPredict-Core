import torch
import joblib
from src.inference_utils import HeteroFTAGNN_v3, compute_period_vol

def load_gnn_model(model_path, in_fm, in_sm, in_tp, in_gf, device='cpu'):
    model = HeteroFTAGNN_v3(in_fm, in_sm, in_tp, in_gf)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def load_rf_model(rf_path):
    return joblib.load(rf_path)

def load_scaler(scaler_path):
    return joblib.load(scaler_path)
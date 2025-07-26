import shap
import joblib
import numpy as np
from src.data_utils import load_features

_explainer = None
_rf_model = None
_background_data = None
_feat_cols = None

# Default number of background samples
N_BACKGROUND = 100

def get_rf_model(rf_path):
    global _rf_model
    if _rf_model is None:
        _rf_model = joblib.load(rf_path)
    return _rf_model

def get_background_data(features_csv, scaler_path):
    global _background_data, _feat_cols
    if _background_data is None:
        import joblib
        df, feat_cols = load_features(features_csv)
        _feat_cols = feat_cols
        # Sample N rows for background
        sample = df.sample(min(N_BACKGROUND, len(df)), random_state=42)
        scaler = joblib.load(scaler_path)
        _background_data = scaler.transform(sample[feat_cols].values)
    return _background_data, _feat_cols

def get_shap_explainer(rf_path, features_csv, scaler_path):
    global _explainer
    if _explainer is None:
        rf = get_rf_model(rf_path)
        background_data, _ = get_background_data(features_csv, scaler_path)
        _explainer = shap.TreeExplainer(rf, background_data)
    return _explainer

def get_shap_values(rf_path, features_csv, scaler_path, X):
    """
    Returns SHAP values for input X (shape: [1, n_features])
    """
    explainer = get_shap_explainer(rf_path, features_csv, scaler_path)
    shap_values = explainer.shap_values(X)
    return shap_values

def get_feat_cols(features_csv):
    global _feat_cols
    if _feat_cols is None:
        _, feat_cols = load_features(features_csv)
        _feat_cols = feat_cols
    return _feat_cols 
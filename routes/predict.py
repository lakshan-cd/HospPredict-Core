from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict
import os
import yaml
import torch
import numpy as np
from src.predict_utils import load_gnn_model, load_rf_model, load_scaler
from src.neo4j_to_pyg import fetch_hotel_graph, load_feat_map
from src.inference_utils import compute_period_vol
from torch.serialization import add_safe_globals
from torch_geometric.data.storage import BaseStorage, NodeStorage, EdgeStorage
import datetime
import os
from neo4j import GraphDatabase
import pandas as pd
from src.shap_utils import get_shap_values, get_feat_cols

# Add safe globals for torch.load (same as training)
torch.serialization.add_safe_globals([BaseStorage, NodeStorage, EdgeStorage])
add_safe_globals([datetime.date])

router = APIRouter()

class PredictRequest(BaseModel):
    company_id: str
    period: str
    feature_changes: Optional[Dict[str, float]] = None

@router.post("/predict")
def predict(req: PredictRequest):
    # Load config
    cfg_path = os.path.join("config", "default.yml")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    features_csv = cfg['data']['features']['feature_wide_edited']
    feat_map, feat_cols = load_feat_map(features_csv)
    GNN_PATH = "models/best_gnn_v4.pth"
    RF_PATH = "models/rf_gnn_ensemble.joblib"
    SCALER_PATH = "models/scaler/tabular_scaler.joblib"
    gid = f"{req.company_id}_{req.period}"

    # Neo4j connection
    neo4j_cfg = cfg['neo4j']
    driver = GraphDatabase.driver(
        os.getenv('NEO4J_URI', neo4j_cfg.get('uri', 'bolt://localhost:7687')),
        auth=(os.getenv('NEO4J_USER', neo4j_cfg.get('user', 'neo4j')), os.getenv('NEO4J_PASSWORD', neo4j_cfg.get('password', 'password')))
    )
    graph = fetch_hotel_graph(driver, req.company_id, req.period, feat_map, feat_cols)

    # Model input dims
    in_fm = graph['FinancialMetric'].x.shape[1]
    in_sm = graph['StockMetric'].x.shape[1]
    in_tp = 32  # Should match training
    in_gf = graph['GraphFeature'].x.shape[1]

    # Load models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gnn = load_gnn_model(GNN_PATH, in_fm, in_sm, in_tp, in_gf, device=device)
    rf = load_rf_model(RF_PATH)
    scaler = load_scaler(SCALER_PATH)

    # Prepare period embedding
    num_T = len(graph['TimePeriod'].periods)
    period_emb = torch.nn.Embedding(num_T, in_tp).to(device)
    graph = graph.to(device)

    # Wide features for this graph
    if gid not in feat_map:
        raise HTTPException(status_code=404, detail=f"Features not found for {gid}")
    wide = feat_map[gid].copy()

    # Save original wide for original prediction
    wide_orig = wide.copy()

    # Apply feature changes if any
    changed_features = {}
    if req.feature_changes:
        for k, v in req.feature_changes.items():
            if k in feat_cols:
                idx = feat_cols.index(k)
                wide[idx] = v
                changed_features[k] = v
            else:
                raise HTTPException(status_code=400, detail=f"Feature {k} not found in available features.")

    # Standard prediction (original)
    with torch.no_grad():
        period_vol = compute_period_vol(graph)
        emb = gnn._embed(graph, period_vol, period_emb).cpu().numpy()
    wide_scaled = scaler.transform([wide_orig])
    X = np.hstack([wide_scaled[0], emb])
    original_risk = float(rf.predict_proba([X])[0, 1])

    # Top N most influential financial metrics (by RF importance)
    importances = rf.feature_importances_[:len(feat_cols)]
    TOP_N = 10
    top_idx = np.argsort(importances)[::-1][:TOP_N]
    top_features = [(feat_cols[i], importances[i]) for i in top_idx]
    top_feature_values = {feat: float(wide_orig[feat_cols.index(feat)]) for feat, _ in top_features}
    top_influential_features = [
        {"feature": feat, "importance": float(imp), "value": top_feature_values[feat]}
        for feat, imp in top_features
    ]

    # Perturbed prediction (if any feature changed)
    if req.feature_changes:
        wide_scaled_perturbed = scaler.transform([wide])
        X_perturbed = np.hstack([wide_scaled_perturbed[0], emb])
        perturbed_risk = float(rf.predict_proba([X_perturbed])[0, 1])
        delta_risk = perturbed_risk - original_risk
        perturbed_features = {feat_cols[i]: float(wide[i]) for i in range(len(feat_cols))}
        changed_features_original = {k: float(wide_orig[feat_cols.index(k)]) for k in req.feature_changes if k in feat_cols}
    else:
        perturbed_risk = None
        delta_risk = None
        perturbed_features = None
        changed_features_original = None

    original_features = {feat_cols[i]: float(wide_orig[i]) for i in range(len(feat_cols))}

    driver.close()

    return {
        "company_id": req.company_id,
        "period": req.period,
        "original_risk": original_risk,
        "perturbed_risk": perturbed_risk,
        "delta_risk": delta_risk,
        "changed_features": changed_features,
        "changed_features_original": changed_features_original,
        "original_features": original_features,
        "perturbed_features": perturbed_features,
        "top_influential_features": top_influential_features
    }

@router.post("/predict/shap")
def predict_shap(req: PredictRequest):
    # Load config
    cfg_path = os.path.join("config", "default.yml")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    features_csv = cfg['data']['features']['feature_wide_edited']
    feat_map, feat_cols = load_feat_map(features_csv)
    GNN_PATH = "models/best_gnn_v4.pth"
    RF_PATH = "models/rf_gnn_ensemble.joblib"
    SCALER_PATH = "models/scaler/tabular_scaler.joblib"
    gid = f"{req.company_id}_{req.period}"

    # Neo4j connection
    neo4j_cfg = cfg['neo4j']
    driver = GraphDatabase.driver(
        os.getenv('NEO4J_URI', neo4j_cfg.get('uri', 'bolt://localhost:7687')),
        auth=(os.getenv('NEO4J_USER', neo4j_cfg.get('user', 'neo4j')), os.getenv('NEO4J_PASSWORD', neo4j_cfg.get('password', 'password')))
    )
    graph = fetch_hotel_graph(driver, req.company_id, req.period, feat_map, feat_cols)

    # Model input dims
    in_fm = graph['FinancialMetric'].x.shape[1]
    in_sm = graph['StockMetric'].x.shape[1]
    in_tp = 32  # Should match training
    in_gf = graph['GraphFeature'].x.shape[1]

    # Load models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gnn = load_gnn_model(GNN_PATH, in_fm, in_sm, in_tp, in_gf, device=device)
    rf = load_rf_model(RF_PATH)
    scaler = load_scaler(SCALER_PATH)

    # Prepare period embedding
    num_T = len(graph['TimePeriod'].periods)
    period_emb = torch.nn.Embedding(num_T, in_tp).to(device)
    graph = graph.to(device)

    # Wide features for this graph
    if gid not in feat_map:
        raise HTTPException(status_code=404, detail=f"Features not found for {gid}")
    wide = feat_map[gid].copy()

    # Save original wide for original prediction
    wide_orig = wide.copy()

    # Apply feature changes if any
    changed_features = {}
    if req.feature_changes:
        for k, v in req.feature_changes.items():
            if k in feat_cols:
                idx = feat_cols.index(k)
                wide[idx] = v
                changed_features[k] = v
            else:
                raise HTTPException(status_code=400, detail=f"Feature {k} not found in available features.")

    # Standard prediction (original)
    with torch.no_grad():
        period_vol = compute_period_vol(graph)
        emb = gnn._embed(graph, period_vol, period_emb).cpu().numpy()
    wide_scaled = scaler.transform([wide_orig])
    X = np.hstack([wide_scaled[0], emb])
    original_risk = float(rf.predict_proba([X])[0, 1])

    # SHAP-based local feature importance (top 5 wide features)
    shap_values = get_shap_values(RF_PATH, features_csv, SCALER_PATH, X.reshape(1, -1))[1][0]  # [1] for positive class
    feat_cols = get_feat_cols(features_csv)
    shap_wide = shap_values[:len(feat_cols)]
    TOP_N = 5
    top_idx = np.argsort(np.abs(shap_wide))[::-1][:TOP_N]
    top_features = [(feat_cols[i], shap_wide[i]) for i in top_idx]
    top_feature_values = {feat: float(wide_orig[feat_cols.index(feat)]) for feat, _ in top_features}
    top_influential_features = [
        {"feature": feat, "shap_value": float(shap_val), "value": top_feature_values[feat]}
        for feat, shap_val in top_features
    ]

    # Perturbed prediction (if any feature changed)
    if req.feature_changes:
        wide_scaled_perturbed = scaler.transform([wide])
        X_perturbed = np.hstack([wide_scaled_perturbed[0], emb])
        perturbed_risk = float(rf.predict_proba([X_perturbed])[0, 1])
        delta_risk = perturbed_risk - original_risk
        perturbed_features = {feat_cols[i]: float(wide[i]) for i in range(len(feat_cols))}
        changed_features_original = {k: float(wide_orig[feat_cols.index(k)]) for k in req.feature_changes if k in feat_cols}
    else:
        perturbed_risk = None
        delta_risk = None
        perturbed_features = None
        changed_features_original = None

    original_features = {feat_cols[i]: float(wide_orig[i]) for i in range(len(feat_cols))}

    driver.close()

    return {
        "company_id": req.company_id,
        "period": req.period,
        "original_risk": original_risk,
        "perturbed_risk": perturbed_risk,
        "delta_risk": delta_risk,
        "changed_features": changed_features,
        "changed_features_original": changed_features_original,
        "original_features": original_features,
        "perturbed_features": perturbed_features,
        "top_influential_features": top_influential_features
    }

@router.get("/predict/available_features")
def available_features():
    cfg_path = os.path.join("config", "default.yml")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    features_csv = cfg['data']['features']['feature_wide_edited']
    _, feat_cols = load_feat_map(features_csv)
    return {"features": feat_cols}

@router.get("/predict/company/{company_id}/periods")
def get_company_periods(company_id: str):
    cfg_path = os.path.join("config", "default.yml")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    features_csv = cfg['data']['features']['feature_wide_edited']
    df = pd.read_csv(features_csv)
    periods = df[df["company_id"] == company_id]["period"].unique().tolist()
    return {"company_id": company_id, "periods": periods}
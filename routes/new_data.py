from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
import tempfile
import os
import sys
import base64
import io
from neo4j import GraphDatabase
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing.new_data import (
    preprocess_new_financial_data, 
    preprocess_new_trade_data,
    merge_new_data,
    generate_wide_features_for_new_data,
    append_to_features_wide,
    create_feature_map_for_new_data
)
from src.graph_construction.new_graph import (
    create_graph_for_new_data
)
from src.neo4j_to_pyg import fetch_hotel_graph, load_feat_map
from src.predict_utils import load_gnn_model, load_rf_model, load_scaler
from src.inference_utils import compute_period_vol
import torch
import yaml
from torch_geometric.data.storage import BaseStorage, NodeStorage, EdgeStorage
import datetime
from torch.serialization import add_safe_globals
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

# Add safe globals for torch.load (same as training)
torch.serialization.add_safe_globals([BaseStorage, NodeStorage, EdgeStorage])
add_safe_globals([datetime.date])

router = APIRouter()

class NewDataRequest(BaseModel):
    financial_file_base64: str  # Base64 encoded financial CSV
    trade_file_base64: str      # Base64 encoded trade CSV
    company_id: str
    period: str  # YYYY-MM-DD format
    occupancy_drop: Optional[float] = None

class PredictRequest(BaseModel):
    company_id: str
    period: str

@router.post("/predict-new-data")
async def predict_new_data(request: NewDataRequest):
    """
    Upload new financial and trade data for prediction using base64 encoded files
    """
    try:
        # 1. Decode base64 files and save temporarily
        try:
            # Decode financial file
            financial_data = base64.b64decode(request.financial_file_base64)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_fin:
                temp_fin.write(financial_data)
            financial_path = temp_fin.name
            # Decode trade file
            trade_data = base64.b64decode(request.trade_file_base64)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_trade:
                temp_trade.write(trade_data)
            trade_path = temp_trade.name
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 file data: {str(e)}")
        
        # 2. Preprocess new data
        financial_df = preprocess_new_financial_data(financial_path, request.company_id)
        trade_df = preprocess_new_trade_data(trade_path, request.company_id)
        
        # 3. Merge data
        merged_df = merge_new_data(financial_df, trade_df, request.company_id)
        
        # 4. Create complete graph for new company using ALL provided data
        # This loads all data to Neo4j and creates dynamic relationships (like loader.py + dynamic.py)
        print(f"Creating complete graph with ALL available data for {request.company_id}")
        print(f"Available periods in data: {merged_df['PeriodEnd'].dt.strftime('%Y-%m-%d').unique()}")
        print(f"Total data shape: {merged_df.shape}")
        
        # Create graph using all data (same as training approach)
        graph_path = create_graph_for_new_data(request.company_id, request.period, financial_df, trade_df)
        
        # 5. Generate wide features for the specific target period
        # Filter merged data to include only data up to the target period for feature generation
        target_period = pd.to_datetime(request.period)
        filtered_merged_df = merged_df[merged_df['PeriodEnd'] <= target_period].copy()
        
        print(f"Target period: {request.period}")
        print(f"Filtered data periods for features: {filtered_merged_df['PeriodEnd'].dt.strftime('%Y-%m-%d').unique()}")
        print(f"Filtered data shape: {filtered_merged_df.shape}")
        
        if filtered_merged_df.empty:
            raise HTTPException(status_code=400, detail=f"No data found for period {request.period}")
        
        wide_features = generate_wide_features_for_new_data(filtered_merged_df, request.company_id, request.period)
        if wide_features is None:
            raise HTTPException(status_code=400, detail="Failed to generate features")
        
        # 6. Append to features_wide.csv
        append_to_features_wide(wide_features)
        
        # 7. Load models and make prediction using Neo4j approach (like test_neo4j_to_pyg.py)
        config = yaml.safe_load(open("config/default.yml"))
        
        # Connect to Neo4j
        driver = GraphDatabase.driver(
            os.getenv('NEO4J_URI'),
            auth=(os.getenv('NEO4J_USER'), os.getenv('NEO4J_PASSWORD'))
        )
        
        # Load wide features (like test_neo4j_to_pyg.py)
        features_csv = config['data']['features']['feature_wide_edited']
        # Ensure id column is generated when loading features
        feat_map, feat_cols = load_feat_map(features_csv)
        
        # Fetch graph from Neo4j for the specific period (like test_neo4j_to_pyg.py)
        print(f"Fetching graph from Neo4j for {request.company_id} at {request.period}")
        graph = fetch_hotel_graph(driver, request.company_id, request.period, feat_map, feat_cols)
        
        # Get input dimensions from graph (same as training)
        in_fm = graph['FinancialMetric'].x.shape[1]
        in_sm = graph['StockMetric'].x.shape[1]
        in_tp = 32  # This should match your training (embedding dim for TimePeriod)
        in_gf = graph['GraphFeature'].x.shape[1]
        
        print(f"Model input dimensions: in_fm={in_fm}, in_sm={in_sm}, in_tp={in_tp}, in_gf={in_gf}")
        
        # Load models with correct dimensions
        gnn_model = load_gnn_model(
            "models/best_gnn_v4.pth",
            in_fm=in_fm, in_sm=in_sm, in_tp=in_tp, in_gf=in_gf,
            device='cpu'
        )
        rf_model = load_rf_model("models/rf_gnn_ensemble.joblib")
        scaler = load_scaler("models/scaler/tabular_scaler.joblib")
        
        # Set num_nodes for TimePeriod (fix for compute_period_vol)
        if hasattr(graph['TimePeriod'], 'periods') and graph['TimePeriod'].periods is not None:
            graph['TimePeriod'].num_nodes = len(graph['TimePeriod'].periods)
        
        # Prepare period embedding (as in test_predict_inference.py)
        num_T = len(graph['TimePeriod'].periods)
        period_emb = torch.nn.Embedding(num_T, in_tp)
        
        # After loading feat_map and before lookup
        print("Available feature keys:", list(feat_map.keys())[-10:])  # print last 10 keys for brevity
        print("Looking for:", f"{request.company_id}_{request.period}")
        print("repr company_id:", repr(request.company_id))
        print("repr period:", repr(request.period))

        # Get the graph ID
        graph_id = f"{request.company_id}_{request.period}"
        
        # Get wide features for this specific graph
        if graph_id not in feat_map:
            raise HTTPException(status_code=400, detail=f"Features not found for {graph_id}")
        
        wide_features_tensor = feat_map[graph_id]
        
        # Run GNN to get embedding
        gnn_model.eval()
        with torch.no_grad():
            period_vol = compute_period_vol(graph)
            emb = gnn_model._embed(graph, period_vol, period_emb)
            emb = emb.detach().cpu().numpy()
        
        # Scale wide features and concatenate with GNN embedding
        wide_scaled = scaler.transform([wide_features_tensor])
        print("wide_scaled[0] shape:", wide_scaled[0].shape)
        print("emb shape:", emb.shape)

        wide_vec = wide_scaled[0].reshape(-1)
        emb_vec = emb.reshape(-1)
        X = np.hstack([wide_vec, emb_vec])
        print("X shape:", X.shape)
        
        # Make prediction for NEXT QUARTER volatility risk
        predicted_risk = rf_model.predict_proba([X])[0, 1]
        
        # Close Neo4j connection
        driver.close()
        
        # Handle hypothetical scenario if requested
        scenario_result = None
        if request.occupancy_drop is not None:
            # Create modified graph with reduced volatility (simulating occupancy drop)
            graph_modified = graph.clone()
            
            # Modify volatility_30 in StockMetric nodes (simulate occupancy drop effect)
            if 'StockMetric' in graph_modified and graph_modified['StockMetric'].x.size(0) > 0:
                # Assuming volatility_30 is the first feature in StockMetric
                vol_idx = 0  # Adjust if volatility_30 is not the first feature
                graph_modified['StockMetric'].x[:, vol_idx] *= (1 - request.occupancy_drop / 100)
            
            # Re-run GNN with modified graph
            with torch.no_grad():
                period_vol_mod = compute_period_vol(graph_modified)
                emb_modified = gnn_model._embed(graph_modified, period_vol_mod, period_emb)
                emb_modified = emb_modified.detach().cpu().numpy()
            
            # Make prediction with modified embedding
            X_modified = np.hstack([wide_scaled[0], emb_modified])
            predicted_risk_modified = rf_model.predict_proba([X_modified])[0, 1]
            
            scenario_result = {
                "original_risk": predicted_risk,
                "modified_risk": predicted_risk_modified,
                "risk_change": predicted_risk_modified - predicted_risk,
                "occupancy_drop_percent": request.occupancy_drop
            }
        
        # Clean up temporary files
        os.unlink(financial_path)
        os.unlink(trade_path)
        
        response = {
            "company_id": request.company_id,
            "current_period": request.period,
            "predicted_next_quarter_risk": predicted_risk,
            "message": f"Successfully processed new data and predicted NEXT QUARTER volatility risk for {request.period}"
        }
        
        if scenario_result:
            response["scenario_analysis"] = scenario_result
        
        return response
        
    except Exception as e:
        # Clean up on error
        if 'financial_path' in locals():
            os.unlink(financial_path)
        if 'trade_path' in locals():
            os.unlink(trade_path)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict-risk")
async def predict_risk(request: PredictRequest):
    # Load config
    cfg_path = os.path.join("config", "default.yml")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    features_csv = cfg['data']['features']['feature_wide_edited']
    feat_map, feat_cols = load_feat_map(features_csv)
    GNN_PATH = "models/best_gnn_v4.pth"
    RF_PATH = "models/rf_gnn_ensemble.joblib"
    SCALER_PATH = "models/scaler/tabular_scaler.joblib"
    gid = f"{request.company_id}_{request.period}"

    # Load a graph
    # For Neo4j connection, use config or env as in test
    neo4j_cfg = cfg['neo4j']

    driver = GraphDatabase.driver(
        os.getenv('NEO4J_URI', neo4j_cfg.get('uri', 'bolt://localhost:7687')),
        auth=(os.getenv('NEO4J_USER', neo4j_cfg.get('user', 'neo4j')), os.getenv('NEO4J_PASSWORD', neo4j_cfg.get('password', 'password')))
    )
    graph = fetch_hotel_graph(driver, request.company_id, request.period, feat_map, feat_cols)

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

    # Standard prediction
    with torch.no_grad():
        period_vol = compute_period_vol(graph)
        emb = gnn._embed(graph, period_vol, period_emb).cpu().numpy()
    wide_scaled = scaler.transform([wide])
    X = np.hstack([wide_scaled[0], emb])
    prob = rf.predict_proba([X])[0, 1]

    wide_orig = wide.copy()

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

    return {"company_id": request.company_id, "period": request.period, "predicted_risk": float(prob), "top_influential_features": top_influential_features}
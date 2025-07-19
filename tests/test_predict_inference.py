import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dotenv import load_dotenv
load_dotenv()

import yaml
import torch
import numpy as np
from neo4j import GraphDatabase
from src.neo4j_to_pyg import load_feat_map, fetch_hotel_graph
from src.inference_utils import HeteroFTAGNN_v3, compute_period_vol
from src.predict_utils import load_gnn_model, load_rf_model, load_scaler

# --- Load config ---
cfg_path = os.path.join("config", "default.yml")
with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)

# --- Neo4j connection ---
neo4j_cfg = cfg['neo4j']
driver = GraphDatabase.driver(
    os.getenv('NEO4J_URI'),
    auth=(os.getenv('NEO4J_USER'), os.getenv('NEO4J_PASSWORD'))
)

# --- Wide features ---
features_csv = cfg['data']['features']['wide_csv']
feat_map, feat_cols = load_feat_map(features_csv)

# --- Model/scaler paths ---
GNN_PATH = "models/best_gnn_v4.pth"
RF_PATH = "models/rf_gnn_ensemble.joblib"
SCALER_PATH = "models/scaler/tabular_scaler.joblib"

# --- Query parameters ---
company_id = "Ceylon_Hotels_Corporation_PLC"
period = "2022-12-31"
gid = f"{company_id}_{period}"

# --- Load a graph ---
graph = fetch_hotel_graph(driver, company_id, period, feat_map, feat_cols)

# --- Model input dims ---
in_fm = graph['FinancialMetric'].x.shape[1]
in_sm = graph['StockMetric'].x.shape[1]
in_tp = 32  # This should match your training (embedding dim for TimePeriod)
in_gf = graph['GraphFeature'].x.shape[1]

# --- Load models ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gnn = load_gnn_model(GNN_PATH, in_fm, in_sm, in_tp, in_gf, device=device)
rf = load_rf_model(RF_PATH)
scaler = load_scaler(SCALER_PATH)

# --- Prepare period embedding ---
num_T = len(graph['TimePeriod'].periods)
period_emb = torch.nn.Embedding(num_T, in_tp).to(device)
graph = graph.to(device)

# --- Wide features for this graph ---
wide = feat_map[gid].copy()  # already numpy array

# --- 1. Standard prediction ---
with torch.no_grad():
    period_vol = compute_period_vol(graph)
    emb = gnn._embed(graph, period_vol, period_emb).cpu().numpy()
wide_scaled = scaler.transform([wide])
X = np.hstack([wide_scaled[0], emb])
prob = rf.predict_proba([X])[0, 1]
print(f"1. Predicted risk for {company_id} at {period}: {prob:.4f}")

# --- 2. What if volatility_30 drops by 10%? ---
graph_vol = graph.clone()
if graph_vol['StockMetric'].x.shape[1] > 0:
    graph_vol['StockMetric'].x[:, 0] *= 0.9  # 0 is volatility_30
with torch.no_grad():
    period_vol = compute_period_vol(graph_vol)
    emb_vol = gnn._embed(graph_vol, period_vol, period_emb).cpu().numpy()
X_vol = np.hstack([wide_scaled[0], emb_vol])
prob_vol = rf.predict_proba([X_vol])[0, 1]
print(f"2. Predicted risk if volatility_30 drops 10%: {prob_vol:.4f} (delta: {prob_vol - prob:.4f})")

# --- 3. What if Revenue in wide features drops by 20%? ---
wide_revenue = wide.copy()
if "Revenue" in feat_cols:
    revenue_idx = feat_cols.index("Revenue")
    wide_revenue[revenue_idx] *= 0.8
    wide_revenue_scaled = scaler.transform([wide_revenue])
else:
    print("Revenue column not found in wide features!")
    wide_revenue_scaled = scaler.transform([wide])  # fallback to original

with torch.no_grad():
    period_vol = compute_period_vol(graph)
    emb_revenue = gnn._embed(graph, period_vol, period_emb).cpu().numpy()
X_revenue = np.hstack([wide_revenue_scaled[0], emb_revenue])
prob_revenue = rf.predict_proba([X_revenue])[0, 1]
print(f"3. Predicted risk if Revenue drops 20%: {prob_revenue:.4f} (delta: {prob_revenue - prob:.4f})")
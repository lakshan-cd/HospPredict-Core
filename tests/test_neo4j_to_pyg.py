import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dotenv import load_dotenv
load_dotenv()

import yaml
import os
from neo4j import GraphDatabase
from src.neo4j_to_pyg import load_feat_map,fetch_hotel_graph


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

# --- Test: fetch a graph for a company and period ---
company_id = "Aitken_Spence_Hotel_Holdings_PLC"
period = "2014-03-31"  # Use a valid period from your data

graph = fetch_hotel_graph(driver, company_id, period, feat_map, feat_cols)

print("Graph object:", graph)
print("FinancialMetric.x shape:", graph['FinancialMetric'].x.shape)
print("StockMetric.x shape:", graph['StockMetric'].x.shape)
print("GraphFeature.x shape:", graph['GraphFeature'].x.shape)
print("TimePeriod.periods:", graph['TimePeriod'].periods)
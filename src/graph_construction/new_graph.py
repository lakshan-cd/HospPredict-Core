import os
import torch
from neo4j import GraphDatabase
from torch_geometric.data import HeteroData
import pandas as pd
import yaml
from dotenv import load_dotenv
from sklearn.preprocessing import RobustScaler

load_dotenv()

def load_config():
    """Load configuration from default.yml"""
    with open("config/default.yml", "r") as f:
        return yaml.safe_load(f)

def get_driver(cfg):
    """Get Neo4j driver using environment variables"""
    return GraphDatabase.driver(
        os.getenv('NEO4J_URI'),
        auth=(os.getenv('NEO4J_USER'), os.getenv('NEO4J_PASSWORD'))
    )

def load_new_financial_to_neo4j(session, company_id, financial_df):
    """
    Load new financial data to Neo4j (without clearing existing data)
    Based on load_financial from loader.py
    """
    # Get numeric columns (same logic as loader.py)
    metric_cols = [
        c for c in financial_df.columns
        if financial_df[c].dtype.kind in 'fi'
           and c not in ('EPS','BasicDilutedEPS')
    ]

    for _, row in financial_df.iterrows():
        params = {
            'company_id': company_id,
            'period':      row.FiscalPeriod,
            'start':       row.PeriodStart.date().isoformat(),
            'end':         row.PeriodEnd.date().isoformat(),
            'metrics': [
                { 'name': c, 'value': float(row[c]) }
                for c in metric_cols
            ]
        }
        session.run("""
        MERGE (h:Hotel {company_id:$company_id})
          ON CREATE SET h.name = $company_id
        MERGE (tp:TimePeriod {period:$period})
          ON CREATE SET tp.start_date = date($start),
                        tp.end_date   = date($end)
        WITH h, tp, $metrics AS mlist
        UNWIND mlist AS m
          MERGE (fm:FinancialMetric {
                   company_id: h.company_id,
                   period:     tp.period,
                   name:       m.name
                 })
          SET fm.value = m.value
        MERGE (h)-[:HAS_FINANCIAL_METRIC]->(fm)
        MERGE (fm)-[:BELONGS_TO_PERIOD]->(tp)
        """, params)

def load_new_stock_to_neo4j(session, company_id, trade_df):
    """
    Load new stock data to Neo4j (without clearing existing data)
    Based on load_stock from loader.py
    """
    trade_df = trade_df.sort_values('trade_date')
    # Build map of all numeric columns per row
    numeric_cols = [c for c in trade_df.columns if c != 'trade_date' and trade_df[c].dtype.kind in 'fi']

    for _, row in trade_df.iterrows():
        date_str = row.trade_date.date().isoformat()
        metrics_map = {c: float(row[c]) for c in numeric_cols}
        params = {
            'company_id': company_id,
            'date':       date_str,
            'metricsMap': metrics_map
        }
        session.run("""
        MERGE (h:Hotel {company_id:$company_id})
        MERGE (tp:TimePeriod {period:$date})
        MERGE (sm:StockMetric {
          company_id:  $company_id,
          trade_date:  date($date)
        })
        // set all trade columns as properties on sm
        SET sm += $metricsMap
        MERGE (h)-[:HAS_STOCK_PERFORMANCE]->(sm)
        MERGE (sm)-[:BELONGS_TO_PERIOD]->(tp)
        """, params)

def create_dynamic_relationships(session, company_id):
    """
    Create dynamic relationships for new data (based on dynamic.py)
    """
    # 1) Seed QoQ changes for key metrics
    metrics = [
        'Revenue','EBITDA','NetIncome','GrossProfit','OperatingIncome',
        'TotalAssets','CurrentAssets','CashAndCashivalents','Inventory',
        'TotalLiabilities','ShareholdersEquity','WorkingCapital',
        'OperatingCF','InvestingCF','FreeCashFlow','Capex',
        'EPS','BasicDilutedEPS','BookValuePerShare','PERatio','PBRatio',
        'CurrentRatio','QuickRatio','ROA','ROE','DebtToEquity'
    ]
    
    for metric in metrics:
        session.run(f"""
        CALL apoc.periodic.iterate(
          '
            MATCH (f1:FinancialMetric {{name: "{metric}", company_id: "{company_id}"}})
            MATCH (f1)-[:BELONGS_TO_PERIOD]->(tp1:TimePeriod)
            MATCH (f2:FinancialMetric {{name: "{metric}", company_id: "{company_id}"}})
            MATCH (f2)-[:BELONGS_TO_PERIOD]->(tp2:TimePeriod)
            WHERE f1.company_id = f2.company_id
              AND tp2.end_date = tp1.end_date + duration({{months:3}})
            RETURN f1, f2
          ',
          '
            MERGE (f1)-[r:QoQ_CHANGE {{metric: "{metric}"}}]->(f2)
            SET r.weight    = (f2.value - f1.value)/f1.value,
                r.createdAt = datetime()
          ',
          {{batchSize:1000, parallel:false}}
        );
        """)

    # 2) Aggregate volatility to quarter
    session.run("""
    MATCH (sm:StockMetric {company_id:$company})
      WHERE sm.volatility_30 IS NOT NULL
    MATCH (sm)-[:BELONGS_TO_PERIOD]->(tp:TimePeriod)
    WITH tp, avg(sm.volatility_30) AS avgVol
    MERGE (tp)-[r:HAS_VOLATILITY]->(iq:IndicatorSummary {
      company_id:$company,
      period:tp.period
    })
      ON CREATE SET iq.type='QuarterlyVol', iq.value=avgVol
      ON MATCH  SET iq.value=avgVol
    SET r.weight    = avgVol,
        r.createdAt = datetime();
    """, {'company': company_id})

    # 3) Link critical quarters and days
    session.run("""
    MATCH (fm:FinancialMetric {name:'critical_quarter', company_id:$company})
      WHERE fm.value = 1
    MATCH (fm)-[:BELONGS_TO_PERIOD]->(tp:TimePeriod)
    MERGE (tp)-[r:CRITICAL_PERIOD]->(fm)
      SET r.weight    = 1.0,
          r.createdAt = datetime()
    """, {'company': company_id})

    session.run("""
    MATCH (sm:StockMetric {company_id:$company})
      WHERE sm.vol_thresh = 1
    MATCH (sm)-[:BELONGS_TO_PERIOD]->(tp:TimePeriod)
    MERGE (tp)-[r:CRITICAL_DAY]->(sm)
      SET r.weight    = sm.volatility_30,
          r.createdAt = datetime()
    """, {'company': company_id})

def create_graph_for_new_data(company_id, period, financial_df, trade_df):
    """
    Create a new graph for new data and save it
    Returns: graph file path
    """
    # 1. Load data to Neo4j (without clearing existing data)
    cfg = load_config()
    driver = get_driver(cfg)
    
    with driver.session() as session:
        # Load financial data
        load_new_financial_to_neo4j(session, company_id, financial_df)
        
        # Load stock data
        load_new_stock_to_neo4j(session, company_id, trade_df)
        
        # Create dynamic relationships
        create_dynamic_relationships(session, company_id)
    
    driver.close()
    
    # 2. Create PyG graph (using your existing neo4j_to_pyg logic)
    # This will use the fetch_hotel_graph function from neo4j_to_pyg.py
    from src.neo4j_to_pyg import fetch_hotel_graph, load_feat_map
    
    # Load feature map
    feat_map, feat_cols = load_feat_map("data/feature_wide/features_wide.csv")
    
    # Fetch graph from Neo4j
    graph = fetch_hotel_graph(driver, company_id, period, feat_map, feat_cols)
    
    # 3. Save graph
    graph_filename = f"{company_id}_{period}.pt"
    graph_path = f"data/forGNN/quarter_graphs/{graph_filename}"
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(graph_path), exist_ok=True)
    
    torch.save({'graph': graph}, graph_path)
    return graph_path

def load_new_graph_features(company_id, period):
    """
    Load graph features for new data
    Returns: feature mapping for the new graph
    """
    # Load the wide features for this company/period
    features_df = pd.read_csv("data/feature_wide/features_wide.csv", parse_dates=['period'])
    
    # Filter for the specific company/period
    mask = (features_df['company_id'] == company_id) & (features_df['period'] == period)
    if not mask.any():
        return None
    
    # Get feature columns (same as your training pipeline)
    feat_cols = [c for c in features_df.columns if c not in ('company_id', 'period', 'id')]
    
    # Scale features (same as your training pipeline)
    scaler = RobustScaler()
    features_scaled = scaler.fit_transform(features_df[mask][feat_cols])
    
    # Create feature mapping
    id_str = f"{company_id}_{period}"
    feat_map = {id_str: torch.tensor(features_scaled[0], dtype=torch.float).unsqueeze(0)}
    
    return feat_map, feat_cols
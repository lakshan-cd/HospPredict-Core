#!/usr/bin/env python3

import os
import yaml
import pandas as pd
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

def load_config():
    cfg_path = os.path.join(
        os.path.dirname(__file__),
        '..', '..',
        'config', 'default.yml'
    )
    return yaml.safe_load(open(cfg_path))

def get_driver(cfg):
    return GraphDatabase.driver(
        os.getenv('NEO4J_URI'),
        auth=(os.getenv('NEO4J_USER'), os.getenv('NEO4J_PASSWORD'))
    )

def clear_existing_data(session):
    """Clear all existing data before loading new data"""
    session.run("MATCH (n) DETACH DELETE n")
    print("✅ Cleared existing data")

def load_financial(session, comp, path):
    """Load quarterly financials as separate nodes."""
    df = pd.read_csv(path, parse_dates=['PeriodStart','PeriodEnd'])
    metric_cols = [
        c for c in df.columns
        if df[c].dtype.kind in 'fi'
           and c not in ('EPS','BasicDilutedEPS')
    ]

    for _, row in df.iterrows():
        params = {
            'company_id': comp,
            'period':      row.FiscalPeriod,
            'start':       row.PeriodStart.date().isoformat(),
            'end':         row.PeriodEnd  .date().isoformat(),
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

def load_stock(session, comp, path):
    """Load daily trades into one StockMetric node per date."""
    df = pd.read_csv(path, parse_dates=['trade_date'])
    df = df.sort_values('trade_date')
    # Build map of all numeric columns per row
    numeric_cols = [c for c in df.columns if c != 'trade_date' and df[c].dtype.kind in 'fi']

    for _, row in df.iterrows():
        date_str = row.trade_date.date().isoformat()
        metrics_map = {c: float(row[c]) for c in numeric_cols}
        params = {
            'company_id': comp,
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

def main():
    cfg     = load_config()
    driver  = get_driver(cfg)
    fin_dir = cfg['data']['processed']['financial']
    trd_dir = cfg['data']['processed']['trades']

    with driver.session() as session:
        clear_existing_data(session)

        # Financial files
        for fn in os.listdir(fin_dir):
            if not fn.endswith('.csv'):
                continue
            comp = fn.split('_financial_')[0]
            print(f"⏳ Loading FIN for {comp}")
            load_financial(session, comp, os.path.join(fin_dir, fn))

        # Stock/trade files
        for fn in os.listdir(trd_dir):
            if not fn.endswith('.csv'):
                continue
            comp = fn.split('_trade_')[0]
            print(f"⏳ Loading STOCK for {comp}")
            load_stock(session, comp, os.path.join(trd_dir, fn))

    driver.close()
    print("✅  All data loaded to Neo4j.")

if __name__ == '__main__':
    main()

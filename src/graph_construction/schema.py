#!/usr/bin/env python3
import yaml, os
from neo4j import GraphDatabase

def load_config():
    path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'default.yml')
    return yaml.safe_load(open(path))

def main():
    cfg = load_config()
    driver = GraphDatabase.driver(cfg['neo4j']['uri'],
                                  auth=(cfg['neo4j']['user'], cfg['neo4j']['password']))
    with driver.session() as sess:
        # Clear existing data first
        sess.run("""
        MATCH (n)
        DETACH DELETE n
        """)
        print("Cleared existing data from database")
        
        # 1) Hotel/Company constraints
        sess.run("""
          CREATE CONSTRAINT IF NOT EXISTS 
            FOR (h:Hotel) REQUIRE h.company_id IS UNIQUE
        """)
        
        # 2) Time Period constraints
        sess.run("""
          CREATE CONSTRAINT IF NOT EXISTS
            FOR (tp:TimePeriod) REQUIRE tp.period IS UNIQUE
        """)
        
        # 3) Financial Metric constraints
        sess.run("""
          CREATE CONSTRAINT IF NOT EXISTS
            FOR (fm:FinancialMetric) 
            REQUIRE (fm.company_id, fm.period, fm.name) IS NODE KEY
        """)
        
        # 4) Stock Metric constraints
        sess.run("""
          CREATE CONSTRAINT IF NOT EXISTS
            FOR (sm:StockMetric)
            REQUIRE (sm.company_id, sm.trade_date) IS NODE KEY
        """)
        
        # Create indexes for better query performance - one at a time
        sess.run("CREATE INDEX IF NOT EXISTS FOR (h:Hotel) ON (h.name)")
        sess.run("CREATE INDEX IF NOT EXISTS FOR (fm:FinancialMetric) ON (fm.name)")
        sess.run("CREATE INDEX IF NOT EXISTS FOR (sm:StockMetric) ON (sm.trade_date)")
        
    driver.close()
    print("✅  Schema and constraints created successfully.")

if __name__ == "__main__":
    main()

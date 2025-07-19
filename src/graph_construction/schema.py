#!/usr/bin/env python3
import yaml, os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

def load_config():
    path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'default.yml')
    return yaml.safe_load(open(path))

def main():
    neo4j_uri = os.getenv('NEO4J_URI')
    neo4j_user = os.getenv('NEO4J_USER')
    neo4j_password = os.getenv('NEO4J_PASSWORD')
    
    # Validate environment variables
    if not neo4j_uri or not neo4j_user or not neo4j_password:
        raise ValueError("Missing required environment variables: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD")
    
    cfg = load_config()
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
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

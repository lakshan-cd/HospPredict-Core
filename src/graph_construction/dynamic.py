#!/usr/bin/env python3
import os, yaml
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

def load_config():
    cfg_path = os.path.join(
        os.path.dirname(__file__),
        '..','..','config','default.yml'
    )
    return yaml.safe_load(open(cfg_path))

def get_driver(cfg):
    return GraphDatabase.driver(
        os.getenv('NEO4J_URI'),
        auth=(os.getenv('NEO4J_USER'), os.getenv('NEO4J_PASSWORD'))
    )

def seed_qoq_change(session, metric):
    """
    Create :QoQ_CHANGE edges between FinancialMetric nodes 
    that belong to TimePeriods exactly 3 months apart.
    """
    session.run(f"""
    CALL apoc.periodic.iterate(
      '
        MATCH (f1:FinancialMetric {{name: "{metric}"}})
        MATCH (f1)-[:BELONGS_TO_PERIOD]->(tp1:TimePeriod)
        MATCH (f2:FinancialMetric {{name: "{metric}"}})
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

def aggregate_volatility_to_quarter(session, company_id):
    """
    For each quarter, average the precomputed volatility_30
    on StockMetric nodes and link it to the TimePeriod.
    """
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

def seed_critical_quarters(session):
    """
    Link every critical FinancialMetric to its TimePeriod via a CRITICAL_PERIOD edge.
    """
    session.run("""
    MATCH (fm:FinancialMetric {name:'critical_quarter'})
      WHERE fm.value = 1
    MATCH (fm)-[:BELONGS_TO_PERIOD]->(tp:TimePeriod)
    MERGE (tp)-[r:CRITICAL_PERIOD]->(fm)
      SET r.weight    = 1.0,
          r.createdAt = datetime()
    """)

def seed_critical_days(session):
    """
    Link every high-volatility StockMetric to its TimePeriod via a CRITICAL_DAY edge.
    """
    session.run("""
    MATCH (sm:StockMetric)
      WHERE sm.vol_thresh = 1
    MATCH (sm)-[:BELONGS_TO_PERIOD]->(tp:TimePeriod)
    MERGE (tp)-[r:CRITICAL_DAY]->(sm)
      SET r.weight    = sm.volatility_30,
          r.createdAt = datetime()
    """)


def main():
    cfg    = load_config()
    driver = get_driver(cfg)
    with driver.session() as sess:
        # 1) Seed QoQ changes for all your key metrics
        metrics = [
          'Revenue','EBITDA','NetIncome','GrossProfit','OperatingIncome',
          'TotalAssets','CurrentAssets','CashAndCashivalents','Inventory',
          'TotalLiabilities','ShareholdersEquity','WorkingCapital',
          'OperatingCF','InvestingCF','FreeCashFlow','Capex',
          'EPS','BasicDilutedEPS','BookValuePerShare','PERatio','PBRatio',
          'CurrentRatio','QuickRatio','ROA','ROE','DebtToEquity'
        ]
        for metric in metrics:
            seed_qoq_change(sess, metric)

        # 2) Aggregate each company's volatility_30 into quarters
        companies = [
          record['company_id']
          for record in sess.run("MATCH (h:Hotel) RETURN h.company_id AS company_id")
        ]
        for comp in companies:
            aggregate_volatility_to_quarter(sess, comp)

        # 3) Link critical quarters and days
        seed_critical_quarters(sess)
        seed_critical_days(sess)    

    driver.close()
    print("✅ Dynamic relationships (QoQ_CHANGE and HAS_VOLATILITY) seeded.")

if __name__ == '__main__':
    main()

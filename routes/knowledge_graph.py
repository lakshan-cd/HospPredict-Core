from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
import yaml
import math
import re
from datetime import datetime

load_dotenv()

router = APIRouter()

def get_driver():
    return GraphDatabase.driver(
        os.getenv('NEO4J_URI'),
        auth=(os.getenv('NEO4J_USER'), os.getenv('NEO4J_PASSWORD'))
    )

def load_config():
    cfg_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'default.yml')
    return yaml.safe_load(open(cfg_path))

def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    else:
        return obj

def convert_to_financial_period(date_str):
    # Expects 'YYYY-MM-DD', returns 'M/D/YYYY'
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return f"{dt.month}/{dt.day}/{dt.year}"

@router.get("/knowledge-graph/company/{company_id}/metrics/timeline")
def metric_timeline(company_id: str, metric: str):
    """Get time series for a metric for a company."""
    driver = get_driver()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (fm:FinancialMetric {company_id: $company_id, name: $metric})-[:BELONGS_TO_PERIOD]->(tp:TimePeriod)
            RETURN tp.period AS period, fm.value AS value
            ORDER BY tp.period
            """,
            {'company_id': company_id, 'metric': metric}
        )
        data = [{'period': r['period'], 'value': r['value']} for r in result]
    driver.close()
    if not data:
        raise HTTPException(status_code=404, detail="No data found.")
    return sanitize_for_json({'company_id': company_id, 'metric': metric, 'timeline': data})

@router.get("/knowledge-graph/company/{company_id}/qoq_changes")
def qoq_changes(company_id: str, metric: str):
    """Get quarter-over-quarter changes for a metric."""
    driver = get_driver()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (f1:FinancialMetric {company_id: $company_id, name: $metric})-[r:QoQ_CHANGE]->(f2:FinancialMetric)
            RETURN f1.period AS from_period, f2.period AS to_period, r.weight AS qoq_change
            ORDER BY f1.period
            """,
            {'company_id': company_id, 'metric': metric}
        )
        data = [{'from_period': r['from_period'], 'to_period': r['to_period'], 'qoq_change': r['qoq_change']} for r in result]
    driver.close()
    if not data:
        raise HTTPException(status_code=404, detail="No data found.")
    return sanitize_for_json({'company_id': company_id, 'metric': metric, 'qoq_changes': data})

@router.get("/knowledge-graph/company/{company_id}/critical_periods")
def critical_periods(company_id: str):
    """Get all critical periods for a company."""
    driver = get_driver()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (tp:TimePeriod)-[:CRITICAL_PERIOD]->(fm:FinancialMetric {company_id: $company_id})
            RETURN tp.period AS period, fm.name AS metric
            ORDER BY tp.period
            """,
            {'company_id': company_id}
        )
        data = [{'period': r['period'], 'metric': r['metric']} for r in result]
    driver.close()
    return sanitize_for_json({'company_id': company_id, 'critical_periods': data})

@router.get("/knowledge-graph/company/{company_id}/volatility_trend")
def volatility_trend(company_id: str):
    """Get quarterly volatility trend for a company."""
    driver = get_driver()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (tp:TimePeriod)-[:HAS_VOLATILITY]->(iq:IndicatorSummary {company_id: $company_id, type: 'QuarterlyVol'})
            RETURN tp.period AS period, iq.value AS volatility
            ORDER BY tp.period
            """,
            {'company_id': company_id}
        )
        data = [{'period': r['period'], 'volatility': r['volatility']} for r in result]
    driver.close()
    return sanitize_for_json({'company_id': company_id, 'volatility_trend': data})

@router.get("/knowledge-graph/compare")
def compare_metric(companies: List[str] = Query(...), period: str = Query(...), metric: str = Query(...)):
    """Compare a metric across companies for a given period."""
    driver = get_driver()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (fm:FinancialMetric {period: $period, name: $metric})
            WHERE fm.company_id IN $companies
            RETURN fm.company_id AS company_id, fm.value AS value
            """,
            {'companies': companies, 'period': period, 'metric': metric}
        )
        data = [{'company_id': r['company_id'], 'value': r['value']} for r in result]
    driver.close()
    return sanitize_for_json({'period': period, 'metric': metric, 'comparison': data})

@router.get("/knowledge-graph/company/{company_id}/events/patterns")
def detect_patterns(company_id: str, min_consecutive: int = 2, qoq_threshold: float = 0.2):
    """Detect patterns: consecutive critical periods, large QoQ changes."""
    driver = get_driver()
    patterns = {}
    period_pattern = re.compile(r'^\\d{4}-\\d{2}$')  # matches 'YYYY-MM'
    with driver.session() as session:
        # Consecutive critical periods
        crit_result = session.run(
            """
            MATCH (tp:TimePeriod)-[:CRITICAL_PERIOD]->(fm:FinancialMetric {company_id: $company_id})
            RETURN tp.period AS period ORDER BY tp.period
            """,
            {'company_id': company_id}
        )
        crit_periods = [r['period'] for r in crit_result]
        # Find runs of consecutive periods
        runs = []
        run = []
        for i, p in enumerate(crit_periods):
            if not period_pattern.match(str(p)):
                # skip invalid period format
                if len(run) >= min_consecutive:
                    runs.append(run)
                run = []
                continue
            if not run:
                run = [p]
            elif period_pattern.match(str(crit_periods[i-1])) and \
                 (int(p[:4])*12+int(p[5:7])) - (int(crit_periods[i-1][:4])*12+int(crit_periods[i-1][5:7])) == 3:
                run.append(p)
            else:
                if len(run) >= min_consecutive:
                    runs.append(run)
                run = [p]
        if len(run) >= min_consecutive:
            runs.append(run)
        patterns['consecutive_critical_periods'] = runs
        # Large QoQ changes
        qoq_result = session.run(
            """
            MATCH (f1:FinancialMetric {company_id: $company_id})-[r:QoQ_CHANGE]->(f2:FinancialMetric)
            WHERE abs(r.weight) >= $qoq_threshold
            RETURN f1.name AS metric, f1.period AS from_period, f2.period AS to_period, r.weight AS qoq_change
            ORDER BY f1.name, f1.period
            """,
            {'company_id': company_id, 'qoq_threshold': qoq_threshold}
        )
        patterns['large_qoq_changes'] = [dict(r) for r in qoq_result]
    driver.close()
    return sanitize_for_json({'company_id': company_id, 'patterns': patterns})

@router.get("/knowledge-graph/company/{company_id}/explain_risk")
def explain_risk(company_id: str, period: str):
    """Explain why a period is risky by tracing contributing metrics and events."""
    driver = get_driver()
    with driver.session() as session:
        # Find critical metrics and events for this period
        crit_metrics = session.run(
            """
            MATCH (tp:TimePeriod {period: $period})-[:CRITICAL_PERIOD]->(fm:FinancialMetric {company_id: $company_id})
            RETURN fm.name AS metric, fm.value AS value
            """,
            {'company_id': company_id, 'period': period}
        )
        crit_metrics = [dict(r) for r in crit_metrics]
        # Find large QoQ changes ending at this period
        qoq_changes = session.run(
            """
            MATCH (f1:FinancialMetric {company_id: $company_id})-[r:QoQ_CHANGE]->(f2:FinancialMetric {period: $period})
            WHERE abs(r.weight) > 0.15
            RETURN f1.name AS metric, r.weight AS qoq_change, f1.period AS from_period
            """,
            {'company_id': company_id, 'period': period}
        )
        qoq_changes = [dict(r) for r in qoq_changes]
    driver.close()
    return sanitize_for_json({
        'company_id': company_id,
        'period': period,
        'critical_metrics': crit_metrics,
        'large_qoq_changes': qoq_changes
    })

@router.get("/knowledge-graph/company/{company_id}/graph_structure")
def graph_structure(company_id: str, period: Optional[str] = None):
    driver = get_driver()
    nodes = []
    edges = []
    node_ids = set()
    edge_ids = set()
    with driver.session() as session:
        if period:
            fin_period = convert_to_financial_period(period)
            # Fetch Hotel, all FinancialMetric, StockMetric, TimePeriod nodes for the period, and all relationships
            result = session.run(
                """
                MATCH (h:Hotel {company_id: $company_id})
                OPTIONAL MATCH (h)-[hfm:HAS_FINANCIAL_METRIC]->(fm:FinancialMetric {period: $fin_period})-[:BELONGS_TO_PERIOD]->(tp:TimePeriod)
                OPTIONAL MATCH (h)-[hsp:HAS_STOCK_PERFORMANCE]->(sm:StockMetric {trade_date: date($period)})-[:BELONGS_TO_PERIOD]->(tp2:TimePeriod)
                RETURN h, id(h) AS h_id, labels(h) AS h_labels,
                       fm, id(fm) AS fm_id, labels(fm) AS fm_labels,
                       tp, id(tp) AS tp_id, labels(tp) AS tp_labels,
                       sm, id(sm) AS sm_id, labels(sm) AS sm_labels,
                       tp2, id(tp2) AS tp2_id, labels(tp2) AS tp2_labels,
                       hfm, hsp
                """,
                {'company_id': company_id, 'period': period, 'fin_period': fin_period}
            )
            for record in result:
                # Hotel node
                if record['h']:
                    h_id = f"{record['h_labels'][0]}:{record['h_id']}"
                    if h_id not in node_ids:
                        h = dict(record['h'])
                        h['id'] = h_id
                        h['type'] = record['h_labels'][0] if record['h_labels'] else 'Node'
                        nodes.append(h)
                        node_ids.add(h_id)
                # FinancialMetric node
                if record['fm']:
                    fm_id = f"{record['fm_labels'][0]}:{record['fm_id']}"
                    if fm_id not in node_ids:
                        fm = dict(record['fm'])
                        fm['id'] = fm_id
                        fm['type'] = record['fm_labels'][0] if record['fm_labels'] else 'Node'
                        nodes.append(fm)
                        node_ids.add(fm_id)
                # StockMetric node
                if record['sm']:
                    sm_id = f"{record['sm_labels'][0]}:{record['sm_id']}"
                    if sm_id not in node_ids:
                        sm = dict(record['sm'])
                        sm['id'] = sm_id
                        sm['type'] = record['sm_labels'][0] if record['sm_labels'] else 'Node'
                        nodes.append(sm)
                        node_ids.add(sm_id)
                # TimePeriod nodes
                if record['tp']:
                    tp_id = f"{record['tp_labels'][0]}:{record['tp_id']}"
                    if tp_id not in node_ids:
                        tp = dict(record['tp'])
                        tp['id'] = tp_id
                        tp['type'] = record['tp_labels'][0] if record['tp_labels'] else 'Node'
                        nodes.append(tp)
                        node_ids.add(tp_id)
                if record['tp2']:
                    tp2_id = f"{record['tp2_labels'][0]}:{record['tp2_id']}"
                    if tp2_id not in node_ids:
                        tp2 = dict(record['tp2'])
                        tp2['id'] = tp2_id
                        tp2['type'] = record['tp2_labels'][0] if record['tp2_labels'] else 'Node'
                        nodes.append(tp2)
                        node_ids.add(tp2_id)
                # Edges
                if record['hfm'] and record['fm']:
                    edge_id = f"{h_id}->{fm_id}:HAS_FINANCIAL_METRIC"
                    if edge_id not in edge_ids:
                        edges.append({'source': h_id, 'target': fm_id, 'type': 'HAS_FINANCIAL_METRIC'})
                        edge_ids.add(edge_id)
                if record['hsp'] and record['sm']:
                    edge_id = f"{h_id}->{sm_id}:HAS_STOCK_PERFORMANCE"
                    if edge_id not in edge_ids:
                        edges.append({'source': h_id, 'target': sm_id, 'type': 'HAS_STOCK_PERFORMANCE'})
                        edge_ids.add(edge_id)
                if record['fm'] and record['tp']:
                    edge_id = f"{fm_id}->{tp_id}:BELONGS_TO_PERIOD"
                    if edge_id not in edge_ids:
                        edges.append({'source': fm_id, 'target': tp_id, 'type': 'BELONGS_TO_PERIOD'})
                        edge_ids.add(edge_id)
                if record['sm'] and record['tp2']:
                    edge_id = f"{sm_id}->{tp2_id}:BELONGS_TO_PERIOD"
                    if edge_id not in edge_ids:
                        edges.append({'source': sm_id, 'target': tp2_id, 'type': 'BELONGS_TO_PERIOD'})
                        edge_ids.add(edge_id)
        else:
            # Fallback: return all metrics for the company (limit for performance)
            result = session.run(
                """
                MATCH (h:Hotel {company_id: $company_id})-[]-(n)
                RETURN id(n) AS node_id, labels(n) AS node_labels, n
                LIMIT 100
                """,
                {'company_id': company_id}
            )
            for record in result:
                n = dict(record['n'])
                n_id = f"{record['node_labels'][0]}:{record['node_id']}"
                n['id'] = n_id
                n['type'] = record['node_labels'][0] if record['node_labels'] else 'Node'
                if n_id not in node_ids:
                    nodes.append(n)
                    node_ids.add(n_id)
    driver.close()
    return sanitize_for_json({'company_id': company_id, 'period': period, 'nodes': nodes, 'edges': edges})

@router.post("/knowledge-graph/company/{company_id}/simulate")
def simulate_change(company_id: str, period: str, metric: str, new_value: float):
    """Simulate a change in a metric and return the predicted effect on risk (stub)."""
    # In a real implementation, this would call the ML pipeline with the perturbed feature.
    # Here, we just return the intended change for demonstration.
    return {
        'company_id': company_id,
        'period': period,
        'metric': metric,
        'new_value': new_value,
        'message': 'Simulation endpoint stub. Integrate with ML pipeline for real prediction.'
    }

@router.get("/knowledge-graph/company/{company_id}/available_metrics")
def available_metrics(company_id: str):
    """Fetch available metrics for a given company from the knowledge graph."""
    driver = get_driver()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (fm:FinancialMetric {company_id: $company_id})
            RETURN DISTINCT fm.name AS metric
            ORDER BY metric
            """,
            {'company_id': company_id}
        )
        metrics = [r['metric'] for r in result]
    driver.close()
    if not metrics:
        raise HTTPException(status_code=404, detail="No metrics found for this company.")
    return sanitize_for_json({'company_id': company_id, 'available_metrics': metrics})

@router.get("/knowledge-graph/company/{company_id}/available_periods")
def available_periods(company_id: str):
    """Fetch all available periods for a given company from FinancialMetric nodes."""
    driver = get_driver()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (fm:FinancialMetric {company_id: $company_id})
            RETURN DISTINCT fm.period AS period
            ORDER BY period
            """,
            {'company_id': company_id}
        )
        periods = [r['period'] for r in result]
    driver.close()
    if not periods:
        raise HTTPException(status_code=404, detail="No periods found for this company.")
    return {'company_id': company_id, 'available_periods': periods}

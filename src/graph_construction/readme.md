# Knowledge Graph Initialization & Data Loader

This README describes the two core scripts for bootstrapping your Neo4j Knowledge Graph:

1. **schema.py** – defines the node labels, uniqueness constraints and indexes.
2. **loader.py** – ingests your preprocessed CSV files into Neo4j, creating nodes and relationships.

---

## Prerequisites

* **Python 3.8+**

* **Neo4j Aura or Self-Hosted Neo4j** (4.4+)

* A virtual environment with:

  ```bash
  pip install pandas pyyaml neo4j
  ```

* Your processed CSVs organized under:

  ```bash
  data/processed/
    ├─ financial/   ← quarterly metrics files
    └─ trades/      ← daily trade files
  ```

* A `config/default.yml` containing:

  ```yaml
  data:
    processed:
      financial: data/processed/financial
      trades:    data/processed/trades

  neo4j:
    uri:      "neo4j+s://<YOUR_AURA_URI>"
    user:     "neo4j"
    password: "<YOUR_PASSWORD>"
  ```

## 1. Schema Definition (schema.py)

This script clears any existing graph, then creates:

* **Uniqueness Constraints**

  * `Hotel.company_id`
  * `TimePeriod.period`
  * `(FinancialMetric.company_id, period, name)` as a node key
  * `(StockMetric.company_id, trade_date)` as a node key
* **Indexes** on lookup properties to speed up queries:

  * `Hotel.name`
  * `FinancialMetric.name`
  * `StockMetric.trade_date`

**How to run**

```bash
python src/graph_construction/schema.py
```

**Effect:** Any existing nodes/relationships are deleted, then the schema is enforced.
**Outcome:** Your graph is now ready to accept data.

## 2. Data Loader (loader.py)

This script reads every CSV in `data/processed/financial` and `data/processed/trades`, and for each company:

1. Wipes the graph (so you can re-run idempotently).

### Financial Metrics

* Reads `*_financial_preprocessed.csv` (one per company).
* For each quarterly row, it:

  * `MERGE`s a `Hotel` node (`company_id`, `name`)
  * `MERGE`s a `TimePeriod` node (`period`, with `start_date` & `end_date`)
  * `UNWIND`s each numeric metric into its own `FinancialMetric` node
  * Links:

    ```cypher
    (Hotel)-[:HAS_FINANCIAL_METRIC]->(FinancialMetric)-[:BELONGS_TO_PERIOD]->(TimePeriod)
    ```

### Stock Metrics

* Reads `*_trade_preprocessed.csv` (one per company).
* For each daily row, it:

  * `MERGE`s the same `Hotel` node
  * `MERGE`s a `TimePeriod` node whose `period` is the trade date
  * `MERGE`s one `StockMetric` node per (`company_id`, `trade_date`)
  * Sets all price/volume columns as properties on that node
  * Links:

    ```cypher
    (Hotel)-[:HAS_STOCK_PERFORMANCE]->(StockMetric)-[:BELONGS_TO_PERIOD]->(TimePeriod)
    ```

**How to run**

```bash
python src/graph_construction/loader.py
```

**Effect:** Your CSV data is transformed into a rich property graph in Neo4j.
**Outcome:** Four node types (`Hotel`, `TimePeriod`, `FinancialMetric`, `StockMetric`) and three relationship types (`HAS_FINANCIAL_METRIC`, `HAS_STOCK_PERFORMANCE`, `BELONGS_TO_PERIOD`) are populated.

## Graph Model Summary

### Nodes

* **`(:Hotel)`**

  * Properties: `company_id` (unique), `name`

* **`(:TimePeriod)`**

  * Properties: `period` (e.g., “Q1-2024” or “2024-06-02”), `start_date`, `end_date`

* **`(:FinancialMetric)`**

  * Properties: `company_id`, `period`, `name` (e.g., “ROA”), `value`

* **`(:StockMetric)`**

  * Properties: `company_id`, `trade_date`, plus all daily price/volume fields

### Relationships

```cypher
(Hotel)-[:HAS_FINANCIAL_METRIC]->(FinancialMetric)-[:BELONGS_TO_PERIOD]->(TimePeriod)
(Hotel)-[:HAS_STOCK_PERFORMANCE]->(StockMetric)-[:BELONGS_TO_PERIOD]->(TimePeriod)
```

* **Temporal relationships:** Metrics link to their `TimePeriod`.
* **Associative relationships:** Companies link to their metrics.

# ETL Preprocessing Project

This project contains the preprocessing pipeline for financial reports and trade data, including dynamic relationship generation in Neo4j.

## Structure

```
etl_preprocessing_project/
│
├── README.md
├── requirements.txt
├── config/
│   └── default.yml
│
├── data/
│   ├── raw/
│   │   ├── financialdata/  # place your raw financial CSV/XLS files here
│   │   └── tradesdata/     # place your raw trade CSV/XLS files here
│   └── processed/
│       ├── financial/
│       ├── trades/
│       └── merged/
│
├── src/
│   ├── __init__.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── financial.py
│   │   ├── trades.py
│   │   └── main.py
│   └── graph_construction/
│       ├── __init__.py
│       ├── loader.py
│       ├── schema.py
│       └── dynamic.py
│
└── scripts/
    ├── run_etl.sh
    └── run_dynamic.sh
```

## Setup

```bash
# Create a virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Unix/MacOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Clearing Data

Before running a fresh pipeline, you may want to clear both processed files and Neo4j database:

1. **Clear Processed Files**
   ```bash
   # On Windows:
   rmdir /s /q data\processed\*
   # On Unix/MacOS:
   rm -rf data/processed/*
   ```

2. **Clear Neo4j Database**
   ```cypher
   // Run these queries in Neo4j Browser to clear all data
   
   // 1. Delete all relationships
   MATCH ()-[r]->() DELETE r;
   
   // 2. Delete all nodes
   MATCH (n) DELETE n;
   
   // 3. Drop all constraints and indexes
   CALL apoc.schema.assert({},{},true);
   ```

## Complete Pipeline Execution

Follow these steps to run the entire pipeline from preprocessing to dynamic relationships:

1. **Clear Existing Data (Optional)**
   ```bash
   # Clear processed data
   rm -rf data/processed/*
   # On Windows:
   rmdir /s /q data\processed\*
   ```

2. **Configure Neo4j**
   - Ensure your Neo4j instance is running
   - Verify Neo4j connection details in `config/default.yml`
   - Make sure APOC plugin is installed in your Neo4j instance

3. **Run Data Preprocessing**
   ```bash
   # Run the ETL preprocessing
   python -m src.preprocessing.main
   ```

4. **Build Graph Schema**
   ```bash
   # Create constraints and indexes
   python -m src.graph_construction.schema
   ```

5. **Load Data into Neo4j**
   ```bash
   # Load processed data into Neo4j
   python -m src.graph_construction.loader
   ```

6. **Generate Dynamic Relationships**
   ```bash
   # Create QoQ changes and volatility indicators
   python -m src.graph_construction.dynamic
   ```

## Verification Queries

After running the pipeline, you can verify the data in Neo4j Browser:

```cypher
// Check QoQ relationships
MATCH (f1:FinancialMetric)-[r:QoQ_CHANGE]->(f2:FinancialMetric)
RETURN f1.name, f1.period, f2.period, r.weight
LIMIT 5;

// Check volatility indicators
MATCH (tp:TimePeriod)-[r:HAS_VOLATILITY]->(iq:IndicatorSummary)
RETURN tp.period, iq.value
ORDER BY tp.period
LIMIT 5;
```

## Troubleshooting

1. If you encounter Neo4j connection issues:
   - Verify your Neo4j credentials in `config/default.yml`
   - Ensure Neo4j instance is running
   - Check if APOC plugin is installed

2. If preprocessing fails:
   - Check raw data format in `data/raw/`
   - Verify file paths in `config/default.yml`
   - Ensure all required Python packages are installed

3. If dynamic relationships fail:
   - Verify Neo4j APOC plugin installation
   - Check if TimePeriod nodes exist
   - Ensure FinancialMetric nodes have valid period values

7. **Generate labels from data**
   ```bash
   # run script file
   python scripts/generate_labels.py

   # So the answer to “why do we need labels in extract_graph.py when we already used them in Colab?” is:

   Because now we’re turning your Neo4j history into a supervised dataset of (graph, label) pairs — one for each quarter snapshot — not just one graph per company.

   That label-map is what stitches “this quarter’s subgraph” to “next quarter’s outcome.” Without it, the extractor would build graphs but wouldn’t know which ones are positive vs. negative examples.
   ```

8. **Generate pt files from neo4j db**
   ```bash
   # install Packages if not
   pip install torch torch-geometric neo4j

   # run script file
   python scripts/extract_graph.py
   ```
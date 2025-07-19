#!/usr/bin/env bash
# run_dynamic.sh

set -euo pipefail
echo "⏳ Seeding dynamic relationships ..."
python3 src/graph_construction/dynamic.py
echo "✅  Done." 
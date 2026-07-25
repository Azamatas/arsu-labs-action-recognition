#!/bin/bash
# actions/run_ruff.sh
# Run Ruff linter and formatter

echo "=== Running Ruff Linter ==="
ruff check .

echo "=== Running Ruff Formatter ==="
ruff format .

echo "=== Auto-fixing issues ==="
ruff check --fix .

echo "=== Done ==="
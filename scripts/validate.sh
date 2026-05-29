#!/bin/bash
set -e

echo "Running lint checks"
flake8 .

echo "Running type checks"
mypy math_utils.py tests/

echo "Running tests"
pytest

echo "Validation complete"

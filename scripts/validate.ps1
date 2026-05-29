Write-Host "Running lint checks"
flake8 .

Write-Host "Running type checks"
mypy math_utils.py tests/

Write-Host "Running tests"
pytest

Write-Host "Validation complete"

"""
Pytest configuration: add project root to path so tests can import app modules.
Tests should be run from project root: pytest tests/ or python -m pytest tests/
"""
import os
import sys

# Project root (parent of tests/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Change to project root so paths like 'all_data_cleaned.csv', 'saved_models/' work
os.chdir(PROJECT_ROOT)

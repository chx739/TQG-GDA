"""Locate the checkout containing this workflow package, including frozen copies."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

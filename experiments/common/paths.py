"""Repository paths shared by experiment entry points."""
from pathlib import Path

def repository_root():
    return Path(__file__).resolve().parents[2]

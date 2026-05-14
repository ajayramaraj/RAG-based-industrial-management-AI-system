"""
Industrial Maintenance Intelligence System
------------------------------------------
Core backend package for:

- CSV ingestion & validation
- Analytics engine
- Vector retrieval (MiniLM + FAISS)
- LLM generation (Ollama)
- Utility helpers

This file initializes the scripts package
and exposes key modules cleanly.
"""

# Import submodules (clean structured access)
from . import csv_loader
from . import analytics
from . import retriever
from . import generator
from . import utils


# Package metadata
__version__ = "1.0.0"
__author__ = "Ajay - Industrial AI"


# Public API (what gets exposed when importing *)
__all__ = [
    "csv_loader",
    "analytics",
    "retriever",
    "generator",
    "utils",
]
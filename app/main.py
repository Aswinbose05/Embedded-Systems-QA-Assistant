import os
import sys

# Ensure RAG folder is in sys.path
rag_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "RAG"))
if rag_path not in sys.path:
    sys.path.insert(0, rag_path)

from app.main import app

__all__ = ["app"]

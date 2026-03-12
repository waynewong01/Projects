#!/usr/bin/env python3
"""
Entry point for the UWB Indoor Positioning System Dashboard.

Usage:
  python -m files.run
  python files/run.py
"""

import logging
import os
import sys
import uvicorn

# Allow running this file directly while keeping package-style imports working.
if __package__ is None or __package__ == "":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from files.config import config
from files.server import app  # noqa: F401 — needed for uvicorn

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    uvicorn.run(
        "files.server:app",
        host=config.server_host,
        port=config.server_port,
        reload=True,
        log_level="info",
    )

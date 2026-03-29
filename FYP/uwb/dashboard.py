"""
Dashboard HTML loader.
"""

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def load_dashboard_html() -> str:
    return Path(__file__).with_name("dashboard.html").read_text(encoding="utf-8")

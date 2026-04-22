"""Legacy compatibility entry point.

The maintained Streamlit dashboard now lives in app.py.
This shim keeps old launch commands from breaking.
"""

import app  # noqa: F401

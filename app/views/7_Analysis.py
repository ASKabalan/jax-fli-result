"""Analysis page — entry point for Streamlit routing.

All logic lives in app/views/_7_Analysis/. This file is kept minimal so
Streamlit can discover it as a page via the numeric prefix convention.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from app.components.styled_container import inject_custom_css

inject_custom_css()
st.title("Analysis")

from app.views._7_Analysis.form import run

run()

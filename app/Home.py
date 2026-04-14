"""Entry point for the jax-fli Launcher GUI — navigation router."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from app.components.styled_container import inject_custom_css

st.set_page_config(page_title="jax-fli Launcher", page_icon="\U0001f680", layout="wide")
inject_custom_css()

scripts_pages = [
    st.Page("views/0_Home.py", title="Home", icon="\U0001f3e0", default=True),
    st.Page("views/1_Simulate.py", title="Simulate", icon="\U0001f680"),
    st.Page("views/2_Samples.py", title="Samples", icon="\U0001f3b2"),
    st.Page("views/3_Infer.py", title="Infer", icon="\U0001f52c"),
    st.Page("views/4_Extract.py", title="Extract", icon="\U0001f4ca"),
    st.Page("views/5_Born_RT.py", title="Born RT", icon="\U0001f52d"),
    st.Page("views/6_Dorian_RT.py", title="Dorian RT", icon="\U0001f30d"),
]

analysis_pages = [
    st.Page("views/7_Analysis.py", title="Analysis", icon="\U0001f4c8"),
]

pg = st.navigation(
    {
        "Scripts": scripts_pages,
        "Analysis": analysis_pages,
    }
)

pg.run()

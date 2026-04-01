"""Home page for the jax-fli Launcher GUI."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from app.components.styled_container import inject_custom_css

st.set_page_config(page_title="jax-fli Launcher", page_icon="\U0001f680", layout="wide")
inject_custom_css()

st.title("jax-fli Launcher")
st.markdown("""
Welcome to the **jax-fli Launcher GUI**. Use the sidebar to navigate to a subcommand page.

Each page mirrors a `python -m launcher <subcommand>` CLI command. Fill in the parameter
forms on the left, and the generated command string will appear at the bottom of the page.

### Available commands

| Page | Command | Description |
|------|---------|-------------|
| **Simulate** | `simulate` | Submit fli-simulate jobs over a mesh x box x cosmology x seed grid |
| **Grid** | `grid` | Submit a single fli-grid job (full parameter-grid exploration) |
| **Samples** | `samples` | Submit fli-samples jobs across chains x batches |
| **Infer** | `infer` | Submit a single fli-infer MCMC inference job |
| **Extract** | `extract` | Submit a fli-extract job to compute chain statistics |
| **Born RT** | `born-rt` | Submit a fli-born-rt Born lensing post-processing job |
| **Dorian RT** | `dorian-rt` | Submit a fli-dorian-rt ray-tracing lensing job (CPU/MPI) |

### Quick start

```bash
streamlit run app/Home.py
```
""")

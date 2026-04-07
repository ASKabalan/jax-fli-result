"""Landing page for the jax-fli Launcher GUI."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from app.components.styled_container import inject_custom_css

inject_custom_css()

st.title("jax-fli Launcher")
st.markdown("""
Welcome to the **jax-fli Launcher GUI**. Use the sidebar to navigate to a subcommand page.

In order to start be sure that you have installed the package with the correct extras

```bash
pip install git+https://github.com/ASKabalan/jax-fli-result.git
```

For raytracing and sampling scripts you need to add

```bash
pip install git+https://github.com/ASKabalan/jax-fli-result.git[raytracing,sampling]
```

### Available commands

| Page | Command | Description |
|------|---------|-------------|
| **Simulate** | `simulate` / `grid` | Submit jobs over a mesh × box × cosmology × seed grid. *Per job* mode (`simulate`) launches one job per combination with optional profiling; *Single job* mode (`grid`) submits one job that loops over all combinations internally |
| **Samples** | `samples` | Generate samples of simulation (unconditioned) can be used to generate mock samples of initial conditions |
| **Infer** | `infer` | Submit a single fli-infer MCMC inference job |
| **Extract** | `extract` | Submit a fli-extract job to compute chain statistics |
| **Born RT** | `born-rt` | Submit a fli-born-rt Born lensing post-processing job |
| **Dorian RT** | `dorian-rt` | Submit a fli-dorian-rt ray-tracing lensing job (CPU/MPI) |

""")

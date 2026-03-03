---
configs:
  - config_name: 01-step_size_cosmology
    data_files: "01-step_size_selection/catalogs/*.parquet"
  - config_name: 01-step_size_perf
    data_files: "01-step_size_selection/perf.csv"
---

# N-Body Simulation Experiments
This repository tracks data, performance metrics, and logs across different simulation runs.

## Data Organization

Each experiment has a dedicated catalog folder containing the parquet files with the data. The `perf.csv` file contains performance metrics for each experiment.
Not all experiments have perf.csv files

Files are stored in hugging face in repo `repo_id = "ASKabalan/jax-fli-experiments"`



## Experiments


### 01 Step size selection

The goal of this experiment to select the optimal size of the time step of the PM solver.

We test with mesh sizes from `512³` to `4096³` and with 5 different time step sizes for each mesh size. The time step sizes are selected such that the total number of steps is in `[5, 10, 20, 30, 40, 50]`.
`
Analysis notebook : [01-step_size.ipynb](https://github.com/ASKabalan/jax-fli-result/blob/main/analysis/01-step_size.ipynb)

config name : `01-step_size_cosmology`

to load 

```python
from datasets import load_dataset
dataset = load_dataset("ASKabalan/jax-fli-experiments", "01-step_size_cosmology")
```
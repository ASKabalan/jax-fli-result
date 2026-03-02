import os
import shutil
from huggingface_hub import HfApi

REPO_ID  = "ASKabalan/jax-fli-experiments"
EXP_NAME = "01-step_size_selection"

api       = HfApi()
here      = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(here, "..", ".."))

# ── Step 0: Restructure local directories to match HF layout ─────────────────
def move_if_needed(src, dst):
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.move(src, dst)
        print(f"Moved {src} → {dst}")
    elif os.path.exists(dst):
        print(f"Already exists: {dst}, skipping move")

move_if_needed(
    src=os.path.join(here, "results", "cosmology_runs"),
    dst=os.path.join(here, "catalogs"),
)
move_if_needed(
    src=os.path.join(here, "runs", "perf"),
    dst=os.path.join(here, "perf_reports"),
)
move_if_needed(
    src=os.path.join(here, "runs", "perf.csv"),
    dst=os.path.join(here, "perf.csv"),
)

# ── Step 1: Conditionally compress LOGS/traces/ → logs.tar.gz ────────────────
traces_dir   = os.path.join(here, "LOGS", "traces")
archive_path = os.path.join(here, "logs.tar.gz")

if os.path.isdir(traces_dir) and not os.path.exists(archive_path):
    shutil.make_archive(
        base_name=os.path.join(here, "logs"),
        format="gztar",
        root_dir=os.path.join(here, "LOGS"),
        base_dir="traces",
    )
    print("Created logs.tar.gz")
else:
    print("Skipping archive: logs.tar.gz already exists or traces/ not found")

# ── Step 2: Upload catalogs ───────────────────────────────────────────────────
api.upload_folder(
    folder_path=os.path.join(here, "catalogs"),
    path_in_repo=f"{EXP_NAME}/catalogs",
    repo_id=REPO_ID,
    repo_type="dataset",
)
print("Uploaded catalogs/")

# ── Step 3: Upload perf_reports ──────────────────────────────────────────────
api.upload_folder(
    folder_path=os.path.join(here, "perf_reports"),
    path_in_repo=f"{EXP_NAME}/perf_reports",
    repo_id=REPO_ID,
    repo_type="dataset",
)
print("Uploaded perf_reports/")

# ── Step 4: Upload perf.csv ───────────────────────────────────────────────────
api.upload_file(
    path_or_fileobj=os.path.join(here, "perf.csv"),
    path_in_repo=f"{EXP_NAME}/perf.csv",
    repo_id=REPO_ID,
    repo_type="dataset",
)
print("Uploaded perf.csv")

# ── Step 5: Upload logs.tar.gz then clean up ─────────────────────────────────
if os.path.exists(archive_path):
    api.upload_file(
        path_or_fileobj=archive_path,
        path_in_repo=f"{EXP_NAME}/logs.tar.gz",
        repo_id=REPO_ID,
        repo_type="dataset",
    )
    print("Uploaded logs.tar.gz")
    os.remove(archive_path)
    print("Removed local logs.tar.gz")

# ── Step 6: Upload README.md (dataset card) ───────────────────────────────────
api.upload_file(
    path_or_fileobj=os.path.join(repo_root, "README.md"),
    path_in_repo="README.md",
    repo_id=REPO_ID,
    repo_type="dataset",
)
print("Uploaded README.md")

print(f"\nDone! View at: https://huggingface.co/datasets/{REPO_ID}")

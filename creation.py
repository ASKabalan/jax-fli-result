from huggingface_hub import HfApi
import os

REPO_ID = "ASKabalan/jax-fli-experiments"
repo_root = os.path.dirname(os.path.abspath(__file__))

api = HfApi()
api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)
print(f"Repository '{REPO_ID}' is ready.")


# ── Upload README.md (dataset card) ───────────────────────────────────
api.upload_file(
    path_or_fileobj=os.path.join(repo_root, "README.md"),
    path_in_repo="README.md",
    repo_id=REPO_ID,
    repo_type="dataset",
)
print("Uploaded README.md")

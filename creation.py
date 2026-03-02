from huggingface_hub import HfApi

REPO_ID = "ASKabalan/jax-fli-experiments"

api = HfApi()
api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)
print(f"Repository '{REPO_ID}' is ready.")

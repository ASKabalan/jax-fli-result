from datasets import get_dataset_config_names, get_dataset_split_names

repo_id = "ASKabalan/jax-fli-experiments"

print(f"Inspecting Hugging Face Repository: {repo_id}")
print("-" * 50)

try:
    # 1. Fetch all available configurations
    configs = get_dataset_config_names(repo_id)

    if not configs:
        print("No configurations found. Check your README.md YAML block.")

    # 2. Loop through each config and fetch its splits
    for config in configs:
        print(f"📦 Configuration: {config}")

        try:
            splits = get_dataset_split_names(repo_id, config_name=config)
            for split in splits:
                print(f"   └── 🗂️ Split: {split}")
        except Exception as e:
            print(f"   └── ⚠️ Could not load splits for this config. Error: {e}")

        print()  # Adds a blank line for readability

except Exception as e:
    print(f"Failed to connect to the repository. Error: {e}")

from huggingface_hub import create_repo

# Buat dataset repository
repo_id = "archiesinaga/bacteria-classification-test-data"

# Optimasi: gunakan parameter minimal yang diperlukan
create_repo(repo_id, repo_type="dataset", exist_ok=True)

print(f"✅ Dataset repository created: https://huggingface.co/datasets/{repo_id}")
from huggingface_hub import HfApi

api = HfApi()
api.upload_file(
    path_or_fileobj="dataset_card_updated.md",
    path_in_repo="README.md",
    repo_id="archiesinaga/bacteria-classification-test-data",
    repo_type="dataset"
)
print("✅ Dataset README updated!")
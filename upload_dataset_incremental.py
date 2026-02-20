from huggingface_hub import HfApi
import os

REPO_ID = "archiesinaga/bacteria-classification-test-data"
DATASET_FOLDER = "test"

def upload_incremental():
    """Upload dataset per subfolder untuk menghindari timeout"""
    api = HfApi()
    
    print("=" * 60)
    print("🤗 Incremental Dataset Upload")
    print("=" * 60)
    
    # Cek folder
    if not os.path.exists(DATASET_FOLDER):
        print(f"❌ Folder '{DATASET_FOLDER}' tidak ditemukan!")
        return
    
    # Get all subfolders (classes)
    subfolders = [f for f in os.listdir(DATASET_FOLDER) 
                  if os.path.isdir(os.path.join(DATASET_FOLDER, f))]
    
    if not subfolders:
        print(f"❌ Tidak ada subfolder di '{DATASET_FOLDER}'")
        return
    
    print(f"\n📂 Found {len(subfolders)} subfolders (classes):")
    for i, folder in enumerate(subfolders, 1):
        folder_path = os.path.join(DATASET_FOLDER, folder)
        file_count = len([f for f in os.listdir(folder_path) 
                         if os.path.isfile(os.path.join(folder_path, f))])
        print(f"   {i}. {folder} ({file_count} files)")
    
    # Upload each subfolder
    total_success = 0
    total_failed = 0
    
    for i, folder in enumerate(subfolders, 1):
        print(f"\n{'='*60}")
        print(f"📤 Uploading {i}/{len(subfolders)}: {folder}")
        print(f"{'='*60}")
        
        folder_path = os.path.join(DATASET_FOLDER, folder)
        
        try:
            api.upload_folder(
                folder_path=folder_path,
                repo_id=REPO_ID,
                repo_type="dataset",
                path_in_repo=f"test/{folder}"
            )
            print(f"✅ {folder} uploaded successfully!")
            total_success += 1
            
        except Exception as e:
            print(f"❌ {folder} upload failed: {e}")
            total_failed += 1
            
            # Ask if want to continue
            response = input("\n⚠️  Continue with next folder? [Y/n]: ")
            if response.lower() == 'n':
                break
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Upload Summary")
    print("=" * 60)
    print(f"✅ Success: {total_success}/{len(subfolders)}")
    print(f"❌ Failed: {total_failed}/{len(subfolders)}")
    
    if total_success > 0:
        print(f"\n🔗 Dataset: https://huggingface.co/datasets/{REPO_ID}")

if __name__ == "__main__":
    upload_incremental()
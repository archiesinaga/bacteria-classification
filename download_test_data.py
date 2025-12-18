"""
Script untuk download test dataset dari Hugging Face
"""

from huggingface_hub import snapshot_download
import os
import sys

def download_test_data():
    """Download test dataset dari Hugging Face"""
    
    print("=" * 60)
    print("🤗 Bacteria Classification - Test Dataset Downloader")
    print("=" * 60)
    
    # Konfigurasi
    repo_id = "archiesinaga/bacteria-classification-test-data"
    local_dir = "./test"
    
    print(f"\n📦 Repository: {repo_id}")
    print(f"📂 Destination: {local_dir}")
    
    # Cek apakah folder sudah ada
    if os.path.exists(local_dir):
        existing_files = len([f for root, dirs, files in os.walk(local_dir) for f in files])
        if existing_files > 0:
            print(f"\n⚠️  Folder '{local_dir}' already exists with {existing_files} files")
            response = input("   Overwrite? [y/N]: ")
            if response.lower() != 'y':
                print("❌ Download cancelled")
                return False
    
    # Create directory
    os.makedirs(local_dir, exist_ok=True)
    
    # Download
    print(f"\n⏳ Downloading test dataset from Hugging Face...")
    print("   (This may take a while depending on dataset size and internet speed)")
    
    try:
        dataset_path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        
        # Count files and calculate size
        file_count = 0
        total_size = 0
        for root, dirs, files in os.walk(local_dir):
            for file in files:
                if not file.startswith('.'):  # Skip hidden files
                    file_count += 1
                    filepath = os.path.join(root, file)
                    total_size += os.path.getsize(filepath)
        
        total_size_mb = total_size / (1024 * 1024)
        
        print("\n" + "=" * 60)
        print("✅ Download Successful!")
        print("=" * 60)
        print(f"\n📍 Location: {os.path.abspath(local_dir)}")
        print(f"📄 Files: {file_count}")
        print(f"💾 Size: {total_size_mb:.2f} MB")
        
        # Show directory structure
        print(f"\n📂 Dataset Structure:")
        classes = [d for d in os.listdir(local_dir) 
                  if os.path.isdir(os.path.join(local_dir, d)) and not d.startswith('.')]
        
        if classes:
            for cls in sorted(classes):
                cls_path = os.path.join(local_dir, cls)
                cls_files = len([f for f in os.listdir(cls_path) 
                               if os.path.isfile(os.path.join(cls_path, f))])
                print(f"   📁 {cls}/ ({cls_files} images)")
        
        print(f"\n💡 You can now run:")
        print(f"   python evaluasi.py")
        print(f"   python prediksi.py")
        
        return True
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ Download Failed!")
        print("=" * 60)
        print(f"\n⚠️  Error: {e}")
        print(f"\n💡 Solutions:")
        print(f"   1. Check your internet connection")
        print(f"   2. Install huggingface_hub: pip install huggingface_hub")
        print(f"   3. Or download manually from:")
        print(f"      https://huggingface.co/datasets/{repo_id}")
        
        return False

if __name__ == "__main__":
    success = download_test_data()
    sys.exit(0 if success else 1)
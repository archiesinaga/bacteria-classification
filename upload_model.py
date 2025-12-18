from huggingface_hub import HfApi, create_repo
import os

# ===== KONFIGURASI =====
REPO_ID = "archiesinaga/bacteria-classification-convnext"
MODEL_FILE = "best_model.keras"  # Sesuaikan dengan nama file Anda
# =======================

def main():
    api = HfApi()
    
    print("🚀 Mulai upload ke Hugging Face...")
    print("=" * 60)
    
    # 1. Buat repository
    print(f"\n📦 Membuat repository '{REPO_ID}'...")
    try:
        create_repo(
            repo_id=REPO_ID,
            repo_type="model",
            exist_ok=True,
            private=False  # Public repository
        )
        print("✅ Repository siap!")
    except Exception as e:
        print(f"⚠️  Repository mungkin sudah ada: {e}")
    
    # 2. Cek file
    print(f"\n📂 Mengecek file '{MODEL_FILE}'...")
    if not os.path.exists(MODEL_FILE):
        print(f"❌ File tidak ditemukan!")
        print(f"📍 Current directory: {os.getcwd()}")
        print("\n📋 File .keras/.h5 yang tersedia:")
        for f in os.listdir("."):
            if f.endswith((".keras", ".h5", ".hdf5")):
                print(f"   - {f}")
        return
    
    file_size = os.path.getsize(MODEL_FILE) / (1024 * 1024)
    print(f"✅ File ditemukan: {file_size:.2f} MB")
    
    # 3. Upload
    print(f"\n⬆️  Upload model ke Hugging Face...")
    print("⏳ Mohon tunggu, ini mungkin memakan waktu...")
    try:
        api.upload_file(
            path_or_fileobj=MODEL_FILE,
            path_in_repo=MODEL_FILE,
            repo_id=REPO_ID,
            repo_type="model",
        )
        print("✅ Upload berhasil!")
    except Exception as e:
        print(f"❌ Upload gagal: {e}")
        return
    
    # 4. Selesai
    print("\n" + "=" * 60)
    print("🎉 BERHASIL!")
    print("=" * 60)
    print(f"\n🔗 Model tersedia di:")
    print(f"   https://huggingface.co/{REPO_ID}")
    print(f"\n💡 Cara download model:")
    print(f"""
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="{REPO_ID}",
    filename="{MODEL_FILE}"
)
    """)

if __name__ == "__main__":
    main()
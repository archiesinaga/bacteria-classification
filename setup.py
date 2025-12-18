"""
Master setup script - Download model dan dataset sekaligus
"""

import sys
import subprocess

def run_script(script_name):
    """Run a Python script and return success status"""
    print(f"\n{'='*60}")
    print(f"Running {script_name}...")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([sys.executable, script_name], check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ {script_name} failed with error code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"❌ {script_name} not found!")
        return False

def main():
    print("=" * 60)
    print("🤗 Bacteria Classification - Complete Setup")
    print("=" * 60)
    print("\nThis script will download:")
    print("  1. Pre-trained model (~XXX MB)")
    print("  2. Test dataset (~XXX MB)")
    
    response = input("\nContinue? [Y/n]: ")
    if response.lower() == 'n':
        print("Setup cancelled")
        return
    
    # Download model
    print("\n" + "="*60)
    print("📥 Step 1/2: Downloading Model")
    print("="*60)
    model_success = run_script("download_model.py")
    
    # Download dataset
    print("\n" + "="*60)
    print("📥 Step 2/2: Downloading Test Dataset")
    print("="*60)
    dataset_success = run_script("download_test_data.py")
    
    # Summary
    print("\n" + "="*60)
    print("📊 Setup Summary")
    print("="*60)
    print(f"Model:   {'✅ Success' if model_success else '❌ Failed'}")
    print(f"Dataset: {'✅ Success' if dataset_success else '❌ Failed'}")
    
    if model_success and dataset_success:
        print("\n🎉 Setup complete! You can now run:")
        print("   python main.py")
        print("   python evaluasi.py")
    else:
        print("\n⚠️  Some downloads failed. Please run scripts individually:")
        if not model_success:
            print("   python download_model.py")
        if not dataset_success:
            print("   python download_test_data.py")

if __name__ == "__main__":
    main()
# generate_class_names.py
import os

test_folder = 'test'
folders = sorted([f for f in os.listdir(test_folder) 
                 if os.path.isdir(os.path.join(test_folder, f))])

print("="*70)
print("COPY CODE INI KE main.py (replace class_names):")
print("="*70)
print("\nclass_names = [")
for folder in folders:
    print(f"    '{folder}',")
print("]")
print(f"\nTotal: {len(folders)} kelas")
print("="*70)

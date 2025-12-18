# check_versions.py
import tensorflow as tf
import sys

print("="*70)
print("INFORMASI SISTEM")
print("="*70)
print(f"Python Version: {sys.version}")
print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version: {tf.keras.__version__}")
print("="*70)
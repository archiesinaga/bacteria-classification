# test_versions.py
import tensorflow as tf
import streamlit as st
import numpy as np
import pandas as pd

print("✅ Versi Library:")
print(f"   TensorFlow: {tf.version.VERSION}")   # FIX
print(f"   Streamlit: {st.__version__}")
print(f"   NumPy: {np.__version__}")
print(f"   Pandas: {pd.__version__}")

# Test TensorFlow GPU (optional)
print(f"\n🖥️  GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")

# test_load_model.py
import tensorflow as tf

print(f"TensorFlow: {tf.__version__}")

try:
    model = tf.keras.models.load_model('best_model.keras')
    print("✅ Model berhasil dimuat!")
    print(f"   Input: {model.input_shape}")
    print(f"   Output: {model.output_shape}")
except Exception as e:
    print(f"❌ Error: {str(e)}")
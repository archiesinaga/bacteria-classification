import tensorflow as tf
import os

class ModelHandler:
    def __init__(self, model_path, class_names):
        """
        Kelas untuk menangani pemuatan dan pengelolaan model.
        
        Args:
            model_path (str): Path ke file model.
            class_names (list): Daftar nama kelas untuk klasifikasi.
        """
        self.model_path = model_path
        self.class_names = class_names
        self.model = None
    
    def load_model(self):
        """
        Memuat model dari file model_path.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"File model tidak ditemukan: {self.model_path}")
        
        self.model = tf.keras.models.load_model(self.model_path)
        print(f"Model berhasil dimuat dari {self.model_path}")
    
    def get_class_names(self):
        """
        Mendapatkan daftar nama kelas.
        
        Returns:
            list: Daftar nama kelas.
        """
        return self.class_names
    
    def predict(self, processed_image):
        """
        Melakukan prediksi pada gambar yang telah diproses.
        
        Args:
            processed_image (numpy.ndarray): Gambar yang telah diproses (4D tensor)
        
        Returns:
            tuple: Label kelas yang diprediksi, probabilitas, dan array prediksi mentah.
        """
        if self.model is None:
            raise ValueError("Model belum dimuat. Gunakan load_model() terlebih dahulu.")
        
        predictions = self.model.predict(processed_image)
        predicted_class = predictions.argmax(axis=1)[0]
        probability = predictions[0][predicted_class]
        
        return self.class_names[predicted_class], probability, predictions
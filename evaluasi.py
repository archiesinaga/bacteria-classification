import numpy as np
import streamlit as st
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
import random
import time
import pandas as pd
import tensorflow as tf

class ModelEvaluation:
    def __init__(self, model_handler, class_names):
        """
        Inisialisasi dengan menggunakan ModelHandler.
        
        Args:
            model_handler (ModelHandler): Objek ModelHandler untuk menangani model.
            class_names (list): Daftar nama kelas untuk evaluasi.
        """
        self.model_handler = model_handler
        self.class_names = class_names
    
    def preprocess_image(self, image_path):
        """
        Preprocess individual image for model input.
        PENTING: Harus sama persis dengan training script (ConvNeXt preprocessing)!
        """
        # 1. Load image
        image = Image.open(image_path)
        
        # 2. Convert ke RGB jika grayscale
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 3. Resize ke ukuran training (224x224)
        image = image.resize((224, 224))
        
        # 4. Convert ke numpy array dengan dtype float32
        image = np.array(image, dtype=np.float32)
        
        # 5. ✅ KUNCI: Gunakan ConvNeXt preprocessing (SAMA dengan training!)
        # IMPORTANT: preprocess_input expects shape (height, width, channels) or (batch, height, width, channels)
        # We have (224, 224, 3), so we need to add batch dimension temporarily
        image_batch = np.expand_dims(image, axis=0)  # Shape: (1, 224, 224, 3)
        image_batch = tf.keras.applications.convnext.preprocess_input(image_batch)
        image = image_batch[0]  # Remove batch dimension: (224, 224, 3)
        
        return image
    
    def select_test_samples(self, class_images, num_samples):
        """Select test samples based on user-specified count with even distribution per class."""
        total_images = sum(len(images) for images in class_images.values())
        
        # Jika jumlah yang diminta sama dengan total data uji, pilih semua data
        if num_samples == total_images:
            sampled_images = [img for images in class_images.values() for img in images]
            sampled_labels = [label for label, images in class_images.items() for _ in images]
            return sampled_images, sampled_labels
        
        # Hitung jumlah gambar per kelas
        images_per_class = num_samples // len(self.class_names)
        extra_images = num_samples % len(self.class_names)
        
        sampled_images, sampled_labels = [], []
        
        for label, images in class_images.items():
            # Tentukan jumlah gambar yang diambil dari setiap kelas
            class_sample_size = images_per_class + (1 if extra_images > 0 else 0)
            extra_images -= 1 if extra_images > 0 else 0
            
            # Pilih gambar secara acak dari kelas
            sampled_class_images = random.sample(
                images, 
                min(class_sample_size, len(images))
            )
            
            # Tambahkan gambar dan label yang terpilih
            sampled_images.extend(sampled_class_images)
            sampled_labels.extend([label] * len(sampled_class_images))
        
        return sampled_images, sampled_labels
    
    def evaluate_model(self, test_images, test_labels):
        """Evaluate model on test images and display results in Streamlit."""
        start_time = time.time()
        
        # Convert list ke numpy array
        test_images = np.array(test_images, dtype=np.float32)
        
        # Validate the shape of the test images array
        if test_images.ndim != 4:
            raise ValueError(f"test_images harus berupa array dengan dimensi (jumlah_gambar, tinggi, lebar, saluran), got shape: {test_images.shape}")
        
        # Prediksi
        predictions = self.model_handler.model.predict(test_images, verbose=1)
        predicted_labels = np.argmax(predictions, axis=1)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Display results
        self.display_evaluation_results(test_images, test_labels, predicted_labels)
        
        return execution_time
    
    def display_evaluation_results(self, test_images, true_labels, predicted_labels):
        """Display evaluation results: prediction images, confusion matrix, and classification report."""
        # Plot predictions using original images (not normalized)
        if hasattr(self, 'original_test_images'):
            self.plot_all_predictions(self.original_test_images, true_labels, predicted_labels, max_cols=10)
        else:
            # Fallback: denormalize if original not available
            self.plot_all_predictions(test_images, true_labels, predicted_labels, max_cols=10)
        
        # Display confusion matrix
        st.write("### Confusion Matrix")
        cm = confusion_matrix(true_labels, predicted_labels)
        
        # Ukuran disesuaikan dengan jumlah kelas
        num_classes = len(self.class_names)
        figsize = (max(16, num_classes * 0.6), max(14, num_classes * 0.5))
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.class_names, 
            yticklabels=self.class_names, 
            ax=ax,
            annot_kws={"size": max(6, 10 - num_classes // 10)}
        )
        ax.set_xlabel('Predicted Labels', fontsize=14)
        ax.set_ylabel('True Labels', fontsize=14)
        ax.set_title('Confusion Matrix', fontsize=18)
        
        # Rotasi label untuk keterbacaan
        plt.xticks(rotation=90, ha='right', fontsize=max(6, 10 - num_classes // 15))
        plt.yticks(rotation=0, fontsize=max(6, 10 - num_classes // 15))
        plt.tight_layout()
        
        st.pyplot(fig)
        
        # Display classification report as a table
        st.write("### Classification Report")
        report = classification_report(
            true_labels, 
            predicted_labels,
            target_names=self.class_names, 
            output_dict=True,
            zero_division=0
        )
        report_df = pd.DataFrame(report).transpose()
        
        # ✅ FIX: Convert support column to object type first before setting string value
        report_df['support'] = report_df['support'].astype('object')
        
        # Format kolom support
        for idx in report_df.index:
            val = report_df.loc[idx, 'support']
            if idx == "accuracy":
                report_df.loc[idx, 'support'] = ""
            elif isinstance(val, (int, float)) and not pd.isna(val):
                report_df.loc[idx, 'support'] = int(val)
        
        # Membulatkan nilai precision, recall, dan f1-score
        report_df[['precision', 'recall', 'f1-score']] = report_df[['precision', 'recall', 'f1-score']].round(2)
        
        # Menampilkan tabel
        st.dataframe(report_df, use_container_width=True, height=min(600, (len(report_df) + 1) * 35))
    
    def plot_all_predictions(self, images, true_labels, predicted_labels, max_cols=10):
        """
        Plot all images with predictions in a compact grid.
        
        Args:
            images (list): Daftar gambar yang diprediksi (bisa original uint8 atau normalized float32).
            true_labels (list): Daftar label sebenarnya.
            predicted_labels (list): Daftar label hasil prediksi.
            max_cols (int): Jumlah kolom maksimum dalam grid.
        """
        num_images = len(images)
        cols = min(max_cols, num_images)
        rows = (num_images // cols) + (num_images % cols > 0)
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        
        # Handle single image case
        if num_images == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, (image, true_label, pred_label) in enumerate(zip(images, true_labels, predicted_labels)):
            ax = axes[i]
            
            # Check if image is already uint8 (original) or needs denormalization
            if image.dtype == np.uint8:
                # Image is already in correct format
                display_image = image
            else:
                # Image needs denormalization
                display_image = image.copy().astype(np.float32)
                
                # ImageNet mean dan std untuk RGB channels
                mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
                std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
                
                # Reshape untuk broadcasting (1, 1, 3)
                mean = mean.reshape(1, 1, 3)
                std = std.reshape(1, 1, 3)
                
                # Denormalisasi
                display_image = (display_image * std) + mean
                
                # Clip dan convert
                display_image = np.clip(display_image, 0, 255).astype(np.uint8)
            
            ax.imshow(display_image)
            ax.axis("off")
            
            # Truncate class name jika terlalu panjang
            true_name = self.class_names[true_label]
            pred_name = self.class_names[pred_label]
            
            if len(true_name) > 15:
                true_name = true_name[:12] + "..."
            if len(pred_name) > 15:
                pred_name = pred_name[:12] + "..."
            
            ax.set_title(
                f"T: {true_name}\nP: {pred_name}",
                color="green" if true_label == pred_label else "red",
                fontsize=7,
                weight='bold' if true_label == pred_label else 'normal'
            )
        
        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")
        
        plt.tight_layout()
        st.pyplot(fig)
    
    def get_test_images(self):
        """Load and preprocess images from the test folder - SIMPLIFIED VERSION."""
        test_folder = os.path.join(os.getcwd(), 'test')
        
        # Cek apakah folder test ada
        if not os.path.exists(test_folder):
            st.error(f"❌ Folder `test/` tidak ditemukan")
            return None, None
        
        test_images, test_labels = [], []
        class_images = {label: [] for label in range(len(self.class_names))}
        original_images = {label: [] for label in range(len(self.class_names))}  # Store original for display
        
        # Load semua gambar dari setiap kelas (tanpa menampilkan detail)
        for label, class_name in enumerate(self.class_names):
            class_folder = os.path.join(test_folder, class_name)
            
            if not os.path.exists(class_folder):
                continue
            
            image_files = [f for f in os.listdir(class_folder) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))]
            
            if len(image_files) == 0:
                continue
            
            for image_file in image_files:
                image_path = os.path.join(class_folder, image_file)
                try:
                    # Load original for visualization
                    original = Image.open(image_path).convert('RGB').resize((224, 224))
                    original_images[label].append(np.array(original, dtype=np.uint8))
                    
                    # Preprocess for model
                    preprocessed = self.preprocess_image(image_path)
                    class_images[label].append(preprocessed)
                except Exception:
                    # Skip gambar yang error tanpa menampilkan warning
                    continue
        
        total_images = sum(len(images) for images in class_images.values())
        
        if total_images == 0:
            st.error("❌ Tidak ada gambar yang ditemukan di folder `test/`")
            return None, None
        
        # Tampilkan info sederhana
        st.success(f"✅ Total {total_images} gambar ditemukan dari {len([v for v in class_images.values() if len(v) > 0])} kelas")
        
        # User input untuk jumlah sampel
        num_to_sample = st.number_input(
            "Jumlah gambar untuk evaluasi:",
            min_value=1,
            max_value=total_images,
            value=min(len(self.class_names) * 10, total_images),
            step=1,
            help="Pilih jumlah gambar yang akan dievaluasi. Gambar akan dipilih secara merata dari setiap kelas."
        )
        
        # Pilih sampel
        test_images, test_labels = self.select_test_samples(class_images, num_to_sample)
        
        # Store original images for visualization
        self.original_test_images, _ = self.select_test_samples(original_images, num_to_sample)
        
        return test_images, test_labels
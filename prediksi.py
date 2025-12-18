import streamlit as st
from PIL import Image
import numpy as np
import time
import tensorflow as tf

class BacteriaClassificationApp:
    def __init__(self, model_handler, class_names):
        self.model_handler = model_handler
        self.class_names = class_names
        self.bacteria_descriptions = {
            "Acinetobacter baumannii": "Bakteri Gram-negatif berbentuk kokus atau kokobasilus, sering menyebabkan infeksi nosokomial yang resisten terhadap antibiotik.",
            "Actinomyces israelii": "Bakteri anaerob Gram-positif yang menyebabkan actinomycosis, sering pada area wajah dan leher.",
            "Bacteroides fragilis": "Bakteri anaerob obligat yang merupakan flora normal usus, dapat menyebabkan infeksi intra-abdomen.",
            "Bifidobacterium spp": "Bakteri Gram-positif anaerob yang merupakan bagian dari flora normal mulut dan usus.",
            "Candida albicans": "Jamur yang dapat menyebabkan kandidiasis pada mulut, kulit, dan area genital.",
            "Clostridium perfringens": "Bakteri anaerob pembentuk spora yang menyebabkan gas gangren dan keracunan makanan.",
            "Enterococcus faecalis": "Bakteri Gram-positif berbentuk kokus, flora normal usus, dapat menyebabkan infeksi saluran kemih.",
            "Enterococcus faecium": "Bakteri Gram-positif yang sering resisten terhadap antibiotik, menyebabkan infeksi nosokomial.",
            "Escherichia coli": "Bakteri Gram-negatif berbentuk batang, flora normal usus, beberapa strain patogenik menyebabkan diare.",
            "Fusobacterium": "Bakteri anaerob Gram-negatif yang terlibat dalam infeksi mulut dan penyakit periodontal.",
            "Lactobacillus casei": "Bakteri probiotik yang membantu kesehatan pencernaan dan sistem imun.",
            "Lactobacillus crispatus": "Bakteri probiotik yang mendominasi flora vagina sehat.",
            "Lactobacillus delbrueckii": "Bakteri yang digunakan dalam produksi yogurt dan produk fermentasi susu.",
            "Lactobacillus gasseri": "Bakteri probiotik yang membantu pencernaan dan kesehatan vagina.",
            "Lactobacillus jensenii": "Bakteri probiotik yang membantu menjaga pH vagina.",
            "Lactobacillus johnsonii": "Bakteri probiotik yang ditemukan di saluran pencernaan manusia.",
            "Lactobacillus paracasei": "Bakteri probiotik yang meningkatkan sistem imun dan kesehatan usus.",
            "Lactobacillus plantarum": "Bakteri probiotik yang ditemukan dalam makanan fermentasi seperti kimchi dan sauerkraut.",
            "Lactobacillus reuteri": "Bakteri probiotik yang membantu kesehatan pencernaan dan sistem imun.",
            "Lactobacillus rhamnosus": "Bakteri probiotik yang membantu mencegah dan mengobati diare.",
            "Lactobacillus salivarius": "Bakteri probiotik yang ditemukan di mulut dan saluran pencernaan.",
            "Listeria monocytogenes": "Bakteri Gram-positif yang menyebabkan listeriosis, berbahaya bagi ibu hamil dan bayi.",
            "Micrococcus spp": "Bakteri Gram-positif yang umumnya tidak patogenik, ditemukan di kulit dan lingkungan.",
            "Neisseria gonorrhoeae": "Bakteri Gram-negatif berbentuk diplokokus yang menyebabkan penyakit gonore.",
            "Porphyromonas gingivalis": "Bakteri anaerob yang menyebabkan penyakit periodontal dan gingivitis.",
            "Propionibacterium acnes": "Bakteri yang terlibat dalam pembentukan jerawat (acne vulgaris).",
            "Proteus": "Bakteri Gram-negatif yang dapat menyebabkan infeksi saluran kemih.",
            "Pseudomonas aeruginosa": "Bakteri Gram-negatif oportunistik yang sering menyebabkan infeksi pada luka bakar.",
            "Staphylococcus aureus": "Bakteri Gram-positif berbentuk kokus, dapat menyebabkan berbagai infeksi dari kulit hingga bakteremia.",
            "Staphylococcus epidermidis": "Bakteri flora normal kulit yang dapat menyebabkan infeksi pada implan medis.",
            "Staphylococcus saprophyticus": "Bakteri yang menyebabkan infeksi saluran kemih, terutama pada wanita muda.",
            "Streptococcus agalactiae": "Bakteri Gram-positif yang dapat menyebabkan infeksi pada bayi baru lahir.",
            "Veillonella": "Bakteri anaerob Gram-negatif yang merupakan flora normal mulut."
        }
    
    def preprocess_image(self, image):
        """
        Preprocessing gambar sesuai dengan training script (ConvNeXt).
        PENTING: Harus sama persis dengan preprocessing saat training!
        """
        # 1. Resize ke ukuran yang sama dengan training (224x224)
        image = image.resize((224, 224))
        
        # 2. Convert ke RGB jika grayscale
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 3. Convert ke numpy array dengan dtype float32
        image = np.array(image, dtype=np.float32)
        
        # 4. Tambahkan batch dimension (1, 224, 224, 3)
        image = np.expand_dims(image, axis=0)
        
        # 5. ✅ KUNCI: Gunakan ConvNeXt preprocessing (SAMA dengan training!)
        # Ini menggunakan ImageNet mean & std normalization
        # Formula: (image - mean) / std
        # mean = [123.675, 116.28, 103.53]
        # std = [58.395, 57.12, 57.375]
        image = tf.keras.applications.convnext.preprocess_input(image)
        
        return image
    
    def predict_bacteria(self, image):
        """Melakukan prediksi jenis bakteri dari citra."""
        start_time = time.time()
        
        # Preprocess gambar (sesuai training)
        processed_image = self.preprocess_image(image)
        
        # Prediksi
        label, probability, _ = self.model_handler.predict(processed_image)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Validasi: deteksi citra bukan bakteri
        if probability < 0.5:
            return None, None, execution_time
        
        return label, probability, execution_time
    
    def display_classification(self):
        """Menampilkan interface upload gambar."""
        uploaded_file = st.file_uploader(
            "Unggah citra mikroskopis bakteri", 
            type=["jpg", "jpeg", "png", "tif", "tiff"]
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Citra mikroskopis yang diunggah', 
                    use_column_width=True)
            return image
        
        return None
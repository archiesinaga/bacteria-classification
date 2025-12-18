# bacteria-classification
Bacteria Classification Task using ConvNext Model
# Bacteria Classification

Deep Learning project untuk klasifikasi bakteri menggunakan ConvNext Model.

## Ì≥ã Deskripsi
Project ini menggunakan Convolutional Neural Network (ConvNext) untuk mengklasifikasikan jenis-jenis bakteri berdasarkan gambar mikroskopis.

## Ì∫Ä Teknologi yang Digunakan
- Python 3.x
- TensorFlow/Keras
- ConvNext Model
- NumPy, Pandas
- Matplotlib

## Ì≥¶ Instalasi

### 1. Clone repository
```bash
git clone https://github.com/archiesinaga/bacteria-classification.git
cd bacteria-classification
```

### 2. Buat virtual environment
```bash
python -m venv venv

# Linux/Mac
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Pre-trained Model Ì¥ó

Model terlalu besar untuk disimpan di GitHub. Download dari Hugging Face:

#### **Opsi A: Download via Python (Recommended)**

Buat file `download_model.py`:
```python
from huggingface_hub import hf_hub_download
import os

print("Ì≥• Downloading model from Hugging Face...")

model_path = hf_hub_download(
    repo_id="archiesinaga/bacteria-classification-convnext",
    filename="best_model.keras",
    local_dir="./models"
)

print(f"‚úÖ Model downloaded successfully!")
print(f"Ì≥ç Model location: {model_path}")
```

Jalankan:
```bash
python download_model.py
```

#### **Opsi B: Download via CLI**
```bash
pip install huggingface_hub
python -m huggingface_hub.commands.huggingface_cli download \
    archiesinaga/bacteria-classification-convnext \
    best_model.keras \
    --local-dir ./models
```

#### **Opsi C: Manual Download**

1. Kunjungi: [https://huggingface.co/archiesinaga/bacteria-classification-convnext](https://huggingface.co/archiesinaga/bacteria-classification-convnext)
2. Klik **"Files and versions"**
3. Download file `best_model.keras`
4. Letakkan di folder `models/`

## Ì≤ª Cara Penggunaan

### Basic Usage
```bash
python main.py
```

### Prediksi Single Image
```bash
python prediksi.py --image path/to/image.jpg
```

### Load Model dalam Code
```python
from tensorflow import keras
from huggingface_hub import hf_hub_download

# Download model jika belum ada
model_path = hf_hub_download(
    repo_id="archiesinaga/bacteria-classification-convnext",
    filename="best_model.keras"
)

# Load model
model = keras.models.load_model(model_path)

# Prediksi
predictions = model.predict(your_image_array)
```

## Ì≥ä Struktur Project
```
bacteria-classification/
‚îú‚îÄ‚îÄ models/
‚îÇ   ‚îú‚îÄ‚îÄ __init__.py
‚îÇ   ‚îú‚îÄ‚îÄ loader.py
‚îÇ   ‚îî‚îÄ‚îÄ best_model.keras      # Download dari Hugging Face
‚îú‚îÄ‚îÄ assets/
‚îú‚îÄ‚îÄ test/
‚îú‚îÄ‚îÄ venv/
‚îú‚îÄ‚îÄ evaluasi.py
‚îú‚îÄ‚îÄ main.py
‚îú‚îÄ‚îÄ prediksi.py
‚îú‚îÄ‚îÄ generate_class_names.py
‚îú‚îÄ‚îÄ check_versions.py
‚îú‚îÄ‚îÄ requirements.txt
‚îî‚îÄ‚îÄ README.md
```

## ÌæØ Model Performance

- **Architecture:** ConvNext
- **Framework:** TensorFlow/Keras
- **Input Size:** 224x224 pixels
- **Classes:** [Jumlah kelas bacteria Anda]

## Ì¥ó Links

- **GitHub Repository:** [bacteria-classification](https://github.com/archiesinaga/bacteria-classification)
- **Hugging Face Model:** [bacteria-classification-convnext](https://huggingface.co/archiesinaga/bacteria-classification-convnext)

## Ì≥à Results

[Tambahkan screenshot hasil klasifikasi, confusion matrix, atau grafik training di sini]

## Ì¥ù Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Ì±§ Author

**Archie Sinaga**

- GitHub: [@archiesinaga](https://github.com/archiesinaga)
- Hugging Face: [@archiesinaga](https://huggingface.co/archiesinaga)

## Ì≥Ñ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Ìπè Acknowledgments

- ConvNext Paper: [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)
- Hugging Face for model hosting
- TensorFlow/Keras team

---

**‚≠ê Jika project ini membantu Anda, berikan star di GitHub!**

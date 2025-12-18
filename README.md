# bacteria-classification
Bacteria Classification Task using ConvNext Model
# Bacteria Classification

Deep Learning project untuk klasifikasi bakteri menggunakan ConvNext Model.

## ��� Deskripsi
Project ini menggunakan Convolutional Neural Network (ConvNext) untuk mengklasifikasikan jenis-jenis bakteri berdasarkan gambar mikroskopis.

## ��� Teknologi yang Digunakan
- Python 3.x
- TensorFlow/Keras
- ConvNext Model
- NumPy, Pandas
- Matplotlib

## ��� Instalasi

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

### 4. Download Pre-trained Model ���

Model terlalu besar untuk disimpan di GitHub. Download dari Hugging Face:

#### **Opsi A: Download via Python (Recommended)**

Buat file `download_model.py`:
```python
from huggingface_hub import hf_hub_download
import os

print("��� Downloading model from Hugging Face...")

model_path = hf_hub_download(
    repo_id="archiesinaga/bacteria-classification-convnext",
    filename="best_model.keras",
    local_dir="./models"
)

print(f"✅ Model downloaded successfully!")
print(f"��� Model location: {model_path}")
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

### 5. Download Test Dataset 📊

**Test dataset is also hosted on Hugging Face**

#### Option A: Automatic Download (Recommended)
```bash
python download_test_data.py
```

#### Option B: Manual Download
```python
from huggingface_hub import snapshot_download

dataset_path = snapshot_download(
    repo_id="archiesinaga/bacteria-classification-test-data",
    repo_type="dataset",
    local_dir="./test"
)
```

#### Option C: Via Browser
1. Visit: [Hugging Face Dataset](https://huggingface.co/datasets/archiesinaga/bacteria-classification-test-data)
2. Download all files
3. Place in `test/` folder

## ��� Cara Penggunaan

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

## ��� Struktur Project
```
bacteria-classification/
├── models/
│   ├── __init__.py
│   ├── loader.py
│   └── best_model.keras      # Download dari Hugging Face
├── assets/
├── test/
├── venv/
├── evaluasi.py
├── main.py
├── prediksi.py
├── generate_class_names.py
├── check_versions.py
├── requirements.txt
└── README.md
```

## ��� Model Performance

- **Architecture:** ConvNext
- **Framework:** TensorFlow/Keras
- **Input Size:** 224x224 pixels
- **Classes:** [33]

## ��� Links

- **GitHub Repository:** [bacteria-classification](https://github.com/archiesinaga/bacteria-classification)
- **Hugging Face Model:** [bacteria-classification-convnext](https://huggingface.co/archiesinaga/bacteria-classification-convnext)

## ��� Results

[]

## ��� Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## ��� Author

**Archie Sinaga**

- GitHub: [@archiesinaga](https://github.com/archiesinaga)
- Hugging Face: [@archiesinaga](https://huggingface.co/archiesinaga)

## ��� License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ��� Acknowledgments

- ConvNext Paper: [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)
- Hugging Face for model hosting
- TensorFlow/Keras team

---

## ⚡ Quick Setup (One Command)

Download both model and dataset:
```bash
python download_model.py && python download_test_data.py
```

**⭐ Jika project ini membantu Anda, berikan star di GitHub!**

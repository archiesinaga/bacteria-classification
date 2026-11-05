# Bacteria Image Classification: ConvNeXt-Tiny vs ResNet-50

Undergraduate thesis (Informatics, Universitas Sriwijaya, 2026). Classifies microscope images of 33 bacteria species and serves the best model in a Streamlit app.

## Question

Can a modern CNN (ConvNeXt-Tiny) beat the classic ResNet-50 baseline at identifying bacteria species from microscope images, and which training setup works best?

## Key results

| Model | Accuracy |
|---|---|
| **ConvNeXt-Tiny** | **99%** (precision, recall and F1 also 99%) |
| ResNet-50 | 98.7% |

- **Data:** DIBaS (Digital Images of Bacteria Species), 2,033 microscope images, 33 species.
- **Setups compared:** 6 training configurations (learning rate x batch size). Best: batch size 32, learning rate 0.001.
- **Test-time augmentation** (averaging predictions over augmented copies of each image) gave 99.1%.

## App

`main.py` is a Streamlit app with two pages:

- **Prediction:** upload a microscope image and get the predicted species with its confidence.
- **Evaluation:** run the model on the test set and see the metrics and confusion matrix.

## Run it locally

```bash
git clone https://github.com/archiesinaga/bacteria-classification.git
cd bacteria-classification
pip install -r requirements.txt
```

The trained model is too large for GitHub and is hosted on Hugging Face. Download it into the repository root, where `main.py` looks for it:

```bash
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('archiesinaga/bacteria-classification-convnext', 'best_model.keras', local_dir='.')"
```

The test images are on Hugging Face too: [archiesinaga/bacteria-classification-test-data](https://huggingface.co/datasets/archiesinaga/bacteria-classification-test-data).

```bash
streamlit run main.py
```

## Repository structure

```
main.py            # Streamlit app: navigation, prediction and evaluation pages
prediksi.py        # image preprocessing and prediction
evaluasi.py        # test-set evaluation, metrics and confusion matrix
models/loader.py   # loads the Keras model
```

## Tech stack

Python · TensorFlow/Keras · ConvNeXt-Tiny · ResNet-50 · scikit-learn · Streamlit · Hugging Face Hub

## Author

**Archie P. H. Sinaga** · Informatics graduate, Universitas Sriwijaya · [GitHub](https://github.com/archiesinaga) · [Portfolio](https://archiesinaga.github.io)

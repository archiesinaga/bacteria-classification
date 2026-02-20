---
license: apache-2.0
task_categories:
- image-classification
tags:
- bacteria
- microscopy
- biology
- medical
size_categories:
- n<1K
---

# Bacteria Classification Test Dataset

Test dataset for bacteria classification using ConvNext model.

## Dataset Description

This dataset contains microscopic images of bacteria for classification tasks.

## Dataset Structure
```
test/
├── class_1/
│   ├── image1.jpg
│   └── ...
├── class_2/
└── ...
```

## Usage

### Quick Download
```bash
pip install huggingface_hub
python download_test_data.py
```

### Download via Python
```python
from huggingface_hub import snapshot_download

dataset_path = snapshot_download(
    repo_id="archiesinaga/bacteria-classification-test-data",
    repo_type="dataset",
    local_dir="./test"
)
print(f"Dataset downloaded to: {dataset_path}")
```

### Download via CLI
```bash
huggingface-cli download \
    archiesinaga/bacteria-classification-test-data \
    --repo-type dataset \
    --local-dir ./test
```

## Use with Project

This dataset is designed to work with the bacteria classification project:
```bash
# Clone project
git clone https://github.com/archiesinaga/bacteria-classification.git
cd bacteria-classification

# Setup environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download model and dataset
python download_model.py
python download_test_data.py

# Run evaluation
python evaluasi.py
```

## Related Resources

- **GitHub Repository**: [bacteria-classification](https://github.com/archiesinaga/bacteria-classification)
- **Pre-trained Model**: [bacteria-classification-convnext](https://huggingface.co/archiesinaga/bacteria-classification-convnext)

## Citation
```bibtex
@dataset{bacteria_test_data_2024,
  author = {Archie Sinaga},
  title = {Bacteria Classification Test Dataset},
  year = {2024},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/datasets/archiesinaga/bacteria-classification-test-data}}
}
```

## License

Apache 2.0

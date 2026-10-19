# Plantar Thermogram Classification (PyTorch)

Deep learning pipeline that classifies plantar (foot sole) thermal images as **diabetic** or **non-diabetic (control)**, using transfer learning with convolutional neural networks.

Thermal imaging of the feet is a non-invasive way to spot temperature changes linked to diabetic foot complications, which can appear before visible ulcers. This project turns that idea into a reproducible, end-to-end PyTorch workflow: data loading, thermal-image preprocessing, training, evaluation and single-image prediction.

> This repository is a clean PyTorch reimplementation of my MSc research on CNN-based plantar thermogram classification, which was published in *Diagnostics* (MDPI, 2023).

## Features

- Transfer learning with ResNet-18, ResNet-50, EfficientNet-B0 or DenseNet-121 (ImageNet weights)
- Subject-level train/validation/test split to prevent data leakage between left and right foot images of the same person
- Thermal-specific augmentation (flips, small rotations, brightness and contrast jitter)
- Class-weighted loss for imbalanced datasets
- Metrics: accuracy, precision, recall, F1, ROC-AUC and confusion matrix, saved to JSON
- Single-image prediction script
- Runs on CPU, a local GPU or free Google Colab GPUs

## Project structure

```
plantar-thermogram-classification/
├── src/
│   ├── dataset.py     # data discovery, subject-level split, transforms
│   ├── model.py       # model factory (transfer learning backbones)
│   ├── train.py       # training loop, early stopping, evaluation
│   └── predict.py     # predict a single thermogram
├── data/              # put the dataset here (not included, see below)
├── outputs/           # checkpoints and metrics are written here
├── requirements.txt
└── README.md
```

## Dataset

The pipeline is designed for the public **Plantar Thermogram Database for the Study of Diabetic Foot Complications** (Hernandez-Contreras et al., IEEE DataPort, 2019), which contains thermograms from a control group and a diabetic (DM) group.

The images are not included in this repository. Download the dataset from IEEE DataPort and arrange it as one folder per class:

```
data/
├── control/
│   ├── CG001_L.png
│   └── ...
└── diabetic/
    ├── DM001_L.png
    └── ...
```

Any folder names work; each sub-folder of `data/` becomes one class. Supported formats: PNG, JPG, BMP, TIF.

## Quick start

```bash
git clone https://github.com/AwaisrazaML/plantar-thermogram-classification.git
cd plantar-thermogram-classification
pip install -r requirements.txt

# train
python src/train.py --data-dir data --model resnet18 --epochs 30

# predict on one image
python src/predict.py --checkpoint outputs/best_model.pt --image path/to/thermogram.png
```

### Subject-level splitting

Each subject usually has more than one image (left and right foot). If images of the same person land in both training and test sets, accuracy looks better than it really is. Pass a regular expression that extracts the subject ID from the file name:

```bash
python src/train.py --data-dir data --subject-regex "^([A-Za-z]+\d+)"
```

With the pattern above, `DM001_L.png` and `DM001_R.png` are kept in the same split.

### Running on Google Colab

1. Open a new notebook and set **Runtime → Change runtime type → GPU**.
2. Clone the repo and install requirements as above.
3. Upload or mount the dataset into `data/`.
4. Run `!python src/train.py --data-dir data --epochs 30`.

## Outputs

After training, `outputs/` contains:

| File | Contents |
|---|---|
| `best_model.pt` | best checkpoint (by validation F1) with class names and settings |
| `metrics.json` | test accuracy, precision, recall, F1, ROC-AUC, confusion matrix |
| `history.json` | per-epoch training and validation loss and F1 |

## Results

Results from this PyTorch reimplementation will be added here after the first full training run.

## Tech stack

Python, PyTorch, torchvision, scikit-learn, NumPy, Pillow

## Author

**Awais Raza**, MSc Electrical Engineering
Research interests: computer vision, medical thermal imaging, explainable AI

## License

MIT

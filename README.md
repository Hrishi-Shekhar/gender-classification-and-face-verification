
# Gender Classification & Face Verification

This repository contains two computer vision pipelines designed for face-based analytics:

- **Task A**: Gender Classification — predict gender from face images.
- **Task B**: Face Verification — match distorted faces to clean reference faces using learned embeddings.

Both tasks are implemented using a combination of pretrained deep learning models and classical ML techniques, with caching and modular structure for fast reproducibility.

## Cloning the Repository

```bash
# First, ensure git-lfs is installed (required for large files like .pth, .npz)
brew install git-lfs        # macOS
sudo apt install git-lfs    # Ubuntu/Debian
choco install git-lfs       # Windows (via Chocolatey)
git lfs install

git clone https://github.com/Hrishi-Shekhar/gender-classification-and-face-verification.git
cd gender-classification-and-face-verification
```

> ⚠️ Note: This repo uses **Git LFS** to manage large binary files like `siamese_model.pth`, `cached_pairs.npz`, and some model weights.  
> If you skip `git lfs install`, **these files may not be fully cloned**, causing broken pipelines.

## Repository Structure

```text
├── task_a.py # Gender classification script
├── task_b.py # Face verification (Siamese network) script
├── cache_taskA/ # Feature and label cache for Task A
├── embed_cache/ # Cached ArcFace embeddings (Task B)
├── saved_models_taskA/ # Trained models (Task A)
├── preprocessed/ # Pickled preprocessed data (Task B)
├── cached_pairs.npz # stores precomputed positive and hard negative embedding pairs
├── siamese_model.pth  # saved PyTorch model file containing the trained Siamese network weights
├── MODEL_DIAGRAMS
├── TechnicalSummary.pdf
├── test.py
├── .gitattributes
├── .gitignore
├── requirements.txt
└── README.md
```

## Task A: Gender Classification

### Goal

Classify face images as **male** or **female** using pretrained CNNs as feature extractors and traditional ML classifiers.

### Features

- Uses `ResNet50`, `EfficientNetB3`, and `VGGFace` (via DeepFace) for feature extraction.
- Features concatenated and passed to:
  - Logistic Regression
  - Support Vector Machine (SVM)
- Automatic caching for speed
- Support for unseen test evaluation

### Expected Dataset Structure

```text
dataset/
├── train/
│ ├── male/
│ └── female/
├── val/
│ ├── male/
│ └── female/
└── test/ (optional)
├── male/
└── female/
```

### Output

Output of metrics on train, val (and optionally test) sets.

## Task B: Face Verification (Siamese Network)

### Goal

Verify if a distorted face belongs to the same identity as a clean reference image. Identities in val/ and test/ are unseen during training.

### Features

1. ArcFace embeddings via DeepFace  
2. Hard negative mining for contrastive learning  
3. Siamese neural network trained on embedding pairs  
4. Caching of faces and embeddings for speed  
5. Evaluation with cosine similarity + thresholding  

### Expected Dataset Structure

```text
dataset/
├── train/
│   └── <person_id>/
│       ├── clean.jpg
│       └── distortion/
│           └── distorted1.jpg
├── val/
│   └── <person_id>/...
└── test/      (optional)
    └── <person_id>/...
```

### Output

Accuracy, Precision, Recall, F1 Score for each of Train/Val/Test.

## Run Instructions

To run both pipelines sequentially using their respective test directories, run:

```bash
python test.py
```

Before running, modify the test_dir paths inside `test.py` to point to your test image directories for:

1. Task A (Gender Classification)
2. Task B (Face Verification)

No need to provide the base dataset directory as cached data is used for both tasks.

## Observational Results

### Task A: Gender Classification

#### Train Set

| Model               | Accuracy | Precision | Recall | F1 Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 1.00     | 1.00      | 1.00   | 1.00     |
| SVM                 | 1.00     | 1.00      | 1.00   | 1.00     |

#### Validation Set

| Model               | Accuracy | Precision | Recall | F1 Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 0.9621   | 0.9619    | 0.9621 | 0.9617   |
| SVM                 | 0.9621   | 0.9620    | 0.9621 | 0.9616   |

---

### Task B: Face Verification

| Dataset    | Accuracy | Precision | Recall | F1 Score |
|------------|----------|-----------|--------|----------|
| Train      | 1.00     | 1.00      | 1.00   | 1.00     |
| Validation | 1.00     | 1.00      | 1.00   | 1.00     |

## Dependencies

Install required packages via:

```bash
pip install -r requirements.txt
```

## Git LFS (Required!)

This project uses Git LFS to store large model and data files:

- `siamese_model.pth` (trained PyTorch model)
- `cached_pairs.npz` (embedding pairs for Task B)
- Any other large cached assets

To set up Git LFS:

```bash
# Install Git LFS
brew install git-lfs        # macOS
sudo apt install git-lfs    # Ubuntu/Debian
choco install git-lfs       # Windows

# Initialize it
git lfs install
```

After installation, **clone the repo again** if you had already cloned it without LFS:

```bash
git clone https://github.com/Hrishi-Shekhar/gender-classification-and-face-verification.git
```

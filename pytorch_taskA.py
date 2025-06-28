import os
import cv2
import joblib
import random
import numpy as np
from tqdm import tqdm
from imutils import paths
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from deepface import DeepFace

# ------------------- Configuration -------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CACHE_DIR = "cache_taskA"
MODELS_DIR = "saved_models_taskA"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ------------------- Preprocessing Functions -------------------
def preprocess_image(image_path, target_size):
    """Read and resize image to target size, then normalize to [0,1]."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    image = cv2.resize(image, target_size)
    image = image.astype("float32") / 255.0
    return image

def cache_path(cache_dir, split_name, name):
    """Helper to create a consistent cache filename."""
    return os.path.join(cache_dir, f"{split_name}_{name}.npy")

def load_and_preprocess_data(image_paths, target_size):
    """Load and preprocess all images, extract labels from folder names."""
    data, labels = [], []
    for image_path in image_paths:
        image = preprocess_image(image_path, target_size)
        label = os.path.basename(os.path.dirname(image_path))
        data.append(image)
        labels.append(label)
    return np.array(data), np.array(labels)

# ------------------- Feature Extraction -------------------
def extract_deepface_features(image_paths, split_name):
    """Extract DeepFace (VGG-Face) embeddings, with caching."""
    cache_file = cache_path(CACHE_DIR, split_name, "deepface")
    if os.path.exists(cache_file):
        return np.load(cache_file)
    features = []
    for path in tqdm(image_paths, desc=f"DeepFace ({split_name})"):
        try:
            rep = DeepFace.represent(img_path=path, model_name='VGG-Face', enforce_detection=False)[0]["embedding"]
            features.append(rep)
        except:
            features.append(np.zeros(2622))  # fallback for failed extraction
    features = np.array(features)
    np.save(cache_file, features)
    return features

def build_torch_model(arch):
    """Initialize pretrained CNN backbone with final pooling removed."""
    base_model = arch(pretrained=True)
    base_model.eval()
    for param in base_model.parameters():
        param.requires_grad = False
    modules = list(base_model.children())[:-1]
    model = nn.Sequential(*modules)
    return model.to(device)

def extract_torch_features(model, images, split_name, name):
    """Extract CNN features using PyTorch model with caching."""
    cache_file = cache_path(CACHE_DIR, split_name, name)
    if os.path.exists(cache_file):
        return np.load(cache_file)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    features = []
    with torch.no_grad():
        for img in tqdm(images, desc=f"Extracting {name}"):
            img_t = transform(img).unsqueeze(0).to(device)
            feat = model(img_t).cpu().numpy().squeeze()
            features.append(feat)
    features = np.array(features)
    np.save(cache_file, features)
    return features

# ------------------- Evaluation -------------------
def evaluate_model(clf, X, y, name):
    """Print standard classification metrics."""
    preds = clf.predict(X)
    print(f"\n{name} Metrics")
    print(f"Accuracy:  {accuracy_score(y, preds):.4f}")
    print(f"Precision: {precision_score(y, preds, average='weighted'):.4f}")
    print(f"Recall:    {recall_score(y, preds, average='weighted'):.4f}")
    print(f"F1 Score:  {f1_score(y, preds, average='weighted'):.4f}")

# ------------------- Main Pipeline -------------------
def run_pipeline(base_dataset_dir, test_dir=None):
    target_size = (224, 224)

    # Step 1: Load image paths for each split
    splits = {
        'train': list(paths.list_images(os.path.join(base_dataset_dir, 'train'))),
        'val': list(paths.list_images(os.path.join(base_dataset_dir, 'val')))
    }
    if test_dir:
        splits['test'] = list(paths.list_images(test_dir))

    # Step 2: Load and preprocess images + extract labels
    images, labels = {}, {}
    for split in splits:
        print(f"[INFO] Loading {split} images: {len(splits[split])}")
        images[split], labels[split] = load_and_preprocess_data(splits[split], target_size)

    # Step 3: Label encoding (use cached encoder if exists)
    le_path = os.path.join(CACHE_DIR, "label_encoder.pkl")
    if os.path.exists(le_path):
        le = joblib.load(le_path)
    else:
        le = LabelEncoder()
        joblib.dump(le.fit(labels['train']), le_path)
    for split in labels:
        labels[split] = le.transform(labels[split])

    # Step 4: Build feature extractors (ResNet + EfficientNet)
    resnet = build_torch_model(models.resnet50)
    efficient = build_torch_model(models.efficientnet_b3)

    # Step 5: Extract and concatenate features from all sources
    features = {}
    for split in images:
        res_feat = extract_torch_features(resnet, images[split], split, 'res')
        eff_feat = extract_torch_features(efficient, images[split], split, 'eff')
        df_feat = extract_deepface_features(splits[split], split)
        features[split] = np.concatenate([res_feat, eff_feat, df_feat], axis=1)

    # Step 6: Normalize features with StandardScaler
    scaler_path = os.path.join(CACHE_DIR, "scaler.pkl")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    else:
        scaler = StandardScaler().fit(features['train'])
        joblib.dump(scaler, scaler_path)
    for split in features:
        features[split] = scaler.transform(features[split])

    # Step 7: Train or load classifiers
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, n_jobs=-1, random_state=SEED),
        "SVM": SVC(kernel='linear', probability=True, random_state=SEED)
    }

    for name, clf in classifiers.items():
        model_path = os.path.join(MODELS_DIR, f"{name.replace(' ', '_')}.pkl")
        if os.path.exists(model_path):
            print(f"\n[INFO] Loading saved model: {name}")
            clf = joblib.load(model_path)
        else:
            print(f"\n[INFO] Training: {name}")
            clf.fit(features['train'], labels['train'])
            joblib.dump(clf, model_path)

        # Step 8: Evaluate model on all splits
        print(f"\n{name} - Training Performance")
        evaluate_model(clf, features['train'], labels['train'], 'Train')

        print(f"\n{name} - Validation Performance")
        evaluate_model(clf, features['val'], labels['val'], 'Val')

        if 'test' in features:
            print(f"\n{name} - Test Performance")
            evaluate_model(clf, features['test'], labels['test'], 'Test')

# ------------------- Entry Point -------------------
if __name__ == '__main__':
    base_dataset_dir = "Comys_Hackathon5 (1)/Comys_Hackathon5/Task_A"
    test_dir = None    # Set test_dir to the path to your test directory or None
    run_pipeline(base_dataset_dir, test_dir)

# face_classification_pipeline.py

import os
import warnings
import cv2
import joblib
import random
import numpy as np
from tqdm import tqdm
from imutils import paths
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from deepface import DeepFace
from transformers.utils import logging as hf_logging

# ------------------- Suppress Warnings -------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*_register_pytree_node is deprecated.*')
warnings.filterwarnings('ignore', message='.*sparse_softmax_cross_entropy is deprecated.*')
hf_logging.set_verbosity_error()

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

# ------------------- Utilities -------------------
def cache_path(name): return os.path.join(CACHE_DIR, name)
def load_pkl(name): return joblib.load(cache_path(name)) if os.path.exists(cache_path(name)) else None
def save_pkl(obj, name): joblib.dump(obj, cache_path(name))
def load_npy(name): return np.load(cache_path(name)) if os.path.exists(cache_path(name)) else None
def save_npy(array, name): np.save(cache_path(name), array)

def preprocess_image(image_path, target_size):
    image = cv2.imread(image_path)
    if image is None: raise ValueError(f"Could not read image: {image_path}")
    image = cv2.resize(image, target_size)
    return image.astype("float32") / 255.0

def load_images_and_labels(image_paths, target_size):
    data, labels = [], []
    for path in image_paths:
        data.append(preprocess_image(path, target_size))
        labels.append(os.path.basename(os.path.dirname(path)))
    return np.array(data), np.array(labels)

def extract_deepface_features(image_paths, split):
    cached = load_npy(f"{split}_deepface.npy")
    if cached is not None: return cached
    features = []
    for path in tqdm(image_paths, desc=f"DeepFace ({split})"):
        try:
            emb = DeepFace.represent(path, model_name='VGG-Face', enforce_detection=False)[0]['embedding']
            features.append(emb)
        except:
            features.append(np.zeros(2622))
    features = np.array(features)
    save_npy(features, f"{split}_deepface.npy")
    return features

def build_torch_model(arch):
    base = arch(pretrained=True)
    base.eval()
    for p in base.parameters(): p.requires_grad = False
    return nn.Sequential(*list(base.children())[:-1]).to(device)

def extract_torch_features(model, images, split, name):
    cached = load_npy(f"{split}_{name}.npy")
    if cached is not None: return cached
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    feats = []
    with torch.no_grad():
        for img in tqdm(images, desc=f"{split} {name}"):
            tensor = transform(img).unsqueeze(0).to(device)
            feat = model(tensor).cpu().numpy().squeeze()
            feats.append(feat)
    feats = np.array(feats)
    save_npy(feats, f"{split}_{name}.npy")
    return feats

def evaluate_model(clf, X, y, name):
    preds = clf.predict(X)
    print(f"\n{name} Metrics")
    print(f"Accuracy:  {accuracy_score(y, preds):.4f}")
    print(f"Precision: {precision_score(y, preds, average='weighted'):.4f}")
    print(f"Recall:    {recall_score(y, preds, average='weighted'):.4f}")
    print(f"F1 Score:  {f1_score(y, preds, average='weighted'):.4f}")

# ------------------- Main Callable -------------------
def run_pipeline(base_dataset_dir=None, test_dir=None):
    target_size = (224, 224)
    resnet = build_torch_model(models.resnet50)
    efficient = build_torch_model(models.efficientnet_b3)

    splits = ['train', 'val']
    images, labels, features = {}, {}, {}

    for split in splits:
        img_file = f"{split}_labels.pkl"
        if base_dataset_dir and not os.path.exists(cache_path(img_file)):
            print(f"[INFO] Processing {split} from images...")
            img_paths = list(paths.list_images(os.path.join(base_dataset_dir, split)))
            X, y = load_images_and_labels(img_paths, target_size)
            save_pkl(img_paths, f"{split}_paths.pkl")
            save_pkl(y, img_file)
            labels[split] = y
            images[split] = X
        else:
            print(f"[INFO] Loading {split} from cache...")
            labels[split] = load_pkl(img_file)
            images[split] = None

    le = load_pkl("label_encoder.pkl")
    if le is None:
        le = LabelEncoder()
        le.fit(labels['train'])
        save_pkl(le, "label_encoder.pkl")
    for split in splits:
        labels[split] = le.transform(labels[split])

    for split in splits:
        if images[split] is not None:
            img_paths = load_pkl(f"{split}_paths.pkl")
            res = extract_torch_features(resnet, images[split], split, 'res')
            eff = extract_torch_features(efficient, images[split], split, 'eff')
            df = extract_deepface_features(img_paths, split)
        else:
            res = load_npy(f"{split}_res.npy")
            eff = load_npy(f"{split}_eff.npy")
            df = load_npy(f"{split}_deepface.npy")
        features[split] = np.concatenate([res, eff, df], axis=1)

    scaler = load_pkl("scaler.pkl")
    if scaler is None:
        scaler = StandardScaler().fit(features['train'])
        save_pkl(scaler, "scaler.pkl")
    for split in splits:
        features[split] = scaler.transform(features[split])

    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, n_jobs=-1, random_state=SEED),
        "SVM": SVC(kernel='linear', probability=True, random_state=SEED)
    }

    for name, clf in classifiers.items():
        path = os.path.join(MODELS_DIR, f"{name.replace(' ', '_')}.pkl")
        if os.path.exists(path):
            print(f"\n[INFO] Loading saved model: {name}")
            clf = joblib.load(path)
        else:
            print(f"\n[INFO] Training: {name}")
            clf.fit(features['train'], labels['train'])
            joblib.dump(clf, path)

        evaluate_model(clf, features['train'], labels['train'], 'Train')
        evaluate_model(clf, features['val'], labels['val'], 'Val')

        if test_dir:
            test_paths = list(paths.list_images(test_dir))
            test_images, test_labels = load_images_and_labels(test_paths, target_size)
            test_labels = le.transform(test_labels)
            res = extract_torch_features(resnet, test_images, 'test', 'res')
            eff = extract_torch_features(efficient, test_images, 'test', 'eff')
            df = extract_deepface_features(test_paths, 'test')
            test_feat = scaler.transform(np.concatenate([res, eff, df], axis=1))
            evaluate_model(clf, test_feat, test_labels, 'Test')

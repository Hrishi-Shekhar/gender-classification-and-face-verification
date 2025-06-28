import os
import cv2
import joblib
import random
import numpy as np
from tqdm import tqdm
from imutils import paths
from sklearn.preprocessing import LabelEncoder, StandardScaler
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.applications import EfficientNetB3, ResNet50
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_efficient
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from deepface import DeepFace
import tensorflow as tf

# ------------------- Configuration -------------------
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Optional: Disable GPU for determinism

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except:
    pass

CACHE_DIR = "cache"
MODELS_DIR = "saved_models"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ------------------- Utility Functions -------------------
def preprocess_image(image_path, target_size):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    image = cv2.resize(image, target_size)
    image = image.astype("float32")
    return image

def cache_path(cache_dir, split_name, name):
    return os.path.join(cache_dir, f"{split_name}_{name}.npy")

def load_and_preprocess_data(image_paths, target_size):
    data, labels = [], []
    for image_path in image_paths:
        image = preprocess_image(image_path, target_size)
        label = os.path.basename(os.path.dirname(image_path))
        data.append(image)
        labels.append(label)
    return np.array(data), np.array(labels)

def extract_deepface_features(image_paths, split_name):
    cache_file = cache_path(CACHE_DIR, split_name, "deepface")
    if os.path.exists(cache_file):
        return np.load(cache_file)
    features = []
    for path in tqdm(image_paths, desc=f"DeepFace ({split_name})"):
        try:
            rep = DeepFace.represent(img_path=path, model_name='VGG-Face', enforce_detection=False)[0]["embedding"]
            features.append(rep)
        except:
            features.append(np.zeros(2622))
    features = np.array(features)
    np.save(cache_file, features)
    return features

def extract_keras_features(model, images, preprocess_func, split_name, name):
    cache_file = cache_path(CACHE_DIR, split_name, name)
    if os.path.exists(cache_file):
        return np.load(cache_file)
    images_pre = preprocess_func(images.copy())
    features = model.predict(images_pre, batch_size=16, verbose=1)
    np.save(cache_file, features)
    return features

def build_model(backbone, preprocess_func, input_shape, name):
    model = backbone(weights='imagenet', include_top=False, input_shape=input_shape)
    model.trainable = False
    out = GlobalAveragePooling2D()(model.output)
    return Model(inputs=model.input, outputs=out), preprocess_func

def evaluate_model(clf, X, y, name):
    preds = clf.predict(X)
    print(f"\n{name} Metrics")
    print(f"Accuracy:  {accuracy_score(y, preds):.4f}")
    print(f"Precision: {precision_score(y, preds, average='weighted'):.4f}")
    print(f"Recall:    {recall_score(y, preds, average='weighted'):.4f}")
    print(f"F1 Score:  {f1_score(y, preds, average='weighted'):.4f}")

def run_pipeline(base_dataset_dir, test_dir=None):
    input_shape = (300, 300, 3)
    target_size = (300, 300)

    splits = {
        'train': list(paths.list_images(os.path.join(base_dataset_dir, 'train'))),
        'val': list(paths.list_images(os.path.join(base_dataset_dir, 'val')))
    }
    if test_dir:
        splits['test'] = list(paths.list_images(test_dir))

    images, labels = {}, {}
    for split in splits:
        print(f"[INFO] Loading {split} images: {len(splits[split])}")
        images[split], labels[split] = load_and_preprocess_data(splits[split], target_size)

    le = LabelEncoder()
    labels['train'] = le.fit_transform(labels['train'])
    labels['val'] = le.transform(labels['val'])
    if 'test' in labels:
        labels['test'] = le.transform(labels['test'])  # Assume label consistency if known

    eff_model, eff_preprocess = build_model(EfficientNetB3, preprocess_efficient, input_shape, "EfficientNetB3")
    res_model, res_preprocess = build_model(ResNet50, preprocess_resnet, input_shape, "ResNet50")

    features = {}
    for split in images:
        eff = extract_keras_features(eff_model, images[split], eff_preprocess, split, 'eff')
        res = extract_keras_features(res_model, images[split], res_preprocess, split, 'res')
        df = extract_deepface_features(splits[split], split)
        features[split] = np.concatenate([eff, res, df], axis=1)

    scaler_path = os.path.join(CACHE_DIR, "scaler.pkl")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    else:
        scaler = StandardScaler().fit(features['train'])
        joblib.dump(scaler, scaler_path)
    for split in features:
        features[split] = scaler.transform(features[split])

    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, n_jobs=-1, random_state=SEED),
        "SVM": SVC(kernel='linear', random_state=SEED)
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for name, clf in classifiers.items():
        print(f"\n\n[INFO] Cross-validating: {name}")
        metrics = defaultdict(list)
        for fold, (train_idx, val_idx) in enumerate(skf.split(features['train'], labels['train']), 1):
            X_train, X_val = features['train'][train_idx], features['train'][val_idx]
            y_train, y_val = labels['train'][train_idx], labels['train'][val_idx]

            clf.fit(X_train, y_train)
            preds = clf.predict(X_val)

            metrics['Accuracy'].append(accuracy_score(y_val, preds))
            metrics['Precision'].append(precision_score(y_val, preds, average='weighted', zero_division=0))
            metrics['Recall'].append(recall_score(y_val, preds, average='weighted', zero_division=0))
            metrics['F1'].append(f1_score(y_val, preds, average='weighted', zero_division=0))

        print(f"{name} - Mean Accuracy:  {np.mean(metrics['Accuracy']):.4f}")
        print(f"{name} - Mean Precision: {np.mean(metrics['Precision']):.4f}")
        print(f"{name} - Mean Recall:    {np.mean(metrics['Recall']):.4f}")
        print(f"{name} - Mean F1 Score:  {np.mean(metrics['F1']):.4f}")

        clf.fit(features['train'], labels['train'])
        joblib.dump(clf, os.path.join(MODELS_DIR, f"{name.replace(' ', '_')}.pkl"))

        print(f"\n{name} - Training Performance")
        evaluate_model(clf, features['train'], labels['train'], 'Train')
        print(f"\n{name} - Validation Performance")
        evaluate_model(clf, features['val'], labels['val'], 'Val')
        if 'test' in features:
            print(f"\n{name} - Test Performance")
            evaluate_model(clf, features['test'], labels['test'], 'Test')


if __name__ == '__main__':
    base_dataset_dir = "Comys_Hackathon5 (1)/Comys_Hackathon5/Task_A"  # Replace with your dataset root
    test_dir = None  # Replace with your test directory or keep None
    run_pipeline(base_dataset_dir, test_dir)

import os
import sys
import joblib
import numpy as np
import cv2
from imutils import paths
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm

# ------------------- Configuration -------------------
CACHE_DIR = "cache_taskA"
MODELS_DIR = "saved_models_taskA"
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------- Utility Functions -------------------
def load_pkl(name):
    path = os.path.join(CACHE_DIR, name)
    if os.path.exists(path):
        return joblib.load(path)
    else:
        raise FileNotFoundError(f"File not found: {path}")

def load_npy(name):
    path = os.path.join(CACHE_DIR, name)
    if os.path.exists(path):
        return np.load(path)
    else:
        raise FileNotFoundError(f"File not found: {path}")

def preprocess_image(image_path, target_size=(224,224)):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    image = cv2.resize(image, target_size)
    image = image.astype("float32") / 255.0
    return image

# ------------------- Torch Model Feature Extractor -------------------
def build_torch_model(arch):
    base = arch(pretrained=True)
    base.eval()
    for p in base.parameters():
        p.requires_grad = False
    # Remove final classifier layer
    return torch.nn.Sequential(*list(base.children())[:-1]).to(DEVICE)

def extract_torch_features(model, images, batch_size=32):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224,224)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    feats = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(images), batch_size), desc="Extracting features"):
            batch_imgs = images[i:i+batch_size]
            batch_tensors = torch.stack([transform(img) for img in batch_imgs]).to(DEVICE)
            batch_feats = model(batch_tensors).cpu().numpy()
            batch_feats = batch_feats.reshape(batch_feats.shape[0], -1)  # flatten output
            feats.append(batch_feats)
    return np.vstack(feats)

# ------------------- Evaluation -------------------
def evaluate_model(clf, X, y):
    preds = clf.predict(X)
    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds, average='weighted')
    rec = recall_score(y, preds, average='weighted')
    f1 = f1_score(y, preds, average='weighted')
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")

# ------------------- Main Inference -------------------
def main(test_dir):
    if not os.path.exists(test_dir):
        raise ValueError(f"Test directory not found: {test_dir}")

    # Load label encoder, scaler, and classifiers
    le = load_pkl("label_encoder.pkl")
    scaler = load_pkl("scaler.pkl")

    clf_lr = joblib.load(os.path.join(MODELS_DIR, "Logistic_Regression.pkl"))
    clf_svm = joblib.load(os.path.join(MODELS_DIR, "SVM.pkl"))

    # Load images and labels from test_dir
    image_paths = list(paths.list_images(test_dir))
    images = [preprocess_image(p) for p in image_paths]
    labels_str = [os.path.basename(os.path.dirname(p)) for p in image_paths]
    labels = le.transform(labels_str)

    # Build torch models for feature extraction
    resnet = build_torch_model(models.resnet50)
    efficient = build_torch_model(models.efficientnet_b3)

    # Extract features
    feats_resnet = extract_torch_features(resnet, images)
    feats_efficient = extract_torch_features(efficient, images)

    # Load cached deepface features for test (assumes cached, else error)
    deepface_feats = load_npy("test_deepface.npy")  # Must be precomputed and cached exactly

    # Combine features and scale
    X_test = np.concatenate([feats_resnet, feats_efficient, deepface_feats], axis=1)
    X_test = scaler.transform(X_test)

    print("\n--- Logistic Regression Test Results ---")
    evaluate_model(clf_lr, X_test, labels)

    print("\n--- SVM Test Results ---")
    evaluate_model(clf_svm, X_test, labels)

# ------------------- Entry -------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python task_a.py <path_to_test_folder>")
        exit(1)
    test_folder = sys.argv[1]
    main(test_folder)

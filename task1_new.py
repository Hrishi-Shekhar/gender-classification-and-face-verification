import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

import cv2
import numpy as np
from imutils import paths
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict
from xgboost import XGBClassifier

from tensorflow.keras.applications import EfficientNetB3, ResNet50
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_efficient
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from deepface import DeepFace
from tqdm import tqdm


# ------------------- Preprocessing -------------------
def preprocess_image(image_path, target_size=(300, 300)):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    image = cv2.resize(image, target_size)
    image = image.astype("float32")
    return image

def load_and_preprocess_data(image_paths, target_size):
    data = []
    labels = []
    for image_path in image_paths:
        image = preprocess_image(image_path, target_size)
        label = os.path.basename(os.path.dirname(image_path))
        data.append(image)
        labels.append(label)
    return np.array(data), np.array(labels)

# ------------------- Data Load -------------------
input_dir = 'Comys_Hackathon5/Comys_Hackathon5/Task_A'
target_size = (300, 300)

train_paths = list(paths.list_images(os.path.join(input_dir, 'train')))
val_paths = list(paths.list_images(os.path.join(input_dir, 'val')))
print(f"[INFO] Found {len(train_paths)} training and {len(val_paths)} validation images.")

trainX, trainY = load_and_preprocess_data(train_paths, target_size)
valX, valY = load_and_preprocess_data(val_paths, target_size)

le = LabelEncoder()
trainY = le.fit_transform(trainY)
valY = le.transform(valY)

# ------------------- Model Loading -------------------
print("[INFO] Loading EfficientNetB3 and ResNet50 models...")
eff_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(300, 300, 3))
eff_model.trainable = False
eff_out = GlobalAveragePooling2D()(eff_model.output)
eff_model = Model(inputs=eff_model.input, outputs=eff_out)

res_model = ResNet50(weights='imagenet', include_top=False, input_shape=(300, 300, 3))
res_model.trainable = False
res_out = GlobalAveragePooling2D()(res_model.output)
res_model = Model(inputs=res_model.input, outputs=res_out)

# ------------------- Feature Extraction -------------------
print("[INFO] Extracting and concatenating features...")
trainX_eff = preprocess_efficient(trainX.copy())
valX_eff = preprocess_efficient(valX.copy())
train_eff_features = eff_model.predict(trainX_eff, batch_size=16, verbose=1)
val_eff_features = eff_model.predict(valX_eff, batch_size=16, verbose=1)

trainX_res = preprocess_resnet(trainX.copy())
valX_res = preprocess_resnet(valX.copy())
train_res_features = res_model.predict(trainX_res, batch_size=16, verbose=1)
val_res_features = res_model.predict(valX_res, batch_size=16, verbose=1)

def extract_deepface_features(image_paths):
    features = []
    for path in tqdm(image_paths, desc="Extracting VGGFace features"):
        try:
            rep = DeepFace.represent(img_path=path, model_name='VGG-Face', enforce_detection=False)[0]["embedding"]
            features.append(rep)
        except Exception as e:
            print(f"[WARNING] Failed to process {path}: {e}")
            features.append(np.zeros(2622))  # fallback to zero vector
    return np.array(features)

# Extract features
train_face_features = extract_deepface_features(train_paths)
val_face_features = extract_deepface_features(val_paths)

train_features = np.concatenate([train_eff_features, train_res_features, train_face_features], axis=1)
val_features = np.concatenate([val_eff_features, val_res_features, val_face_features], axis=1)

# ------------------- Feature Normalization -------------------
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
val_features = scaler.transform(val_features)

# ------------------- Cross-Validation -------------------
print("\n[INFO] Performing 5-Fold Stratified Cross-Validation on Training Data...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=5, min_samples_split=10, min_samples_leaf=4, n_jobs=-1, random_state=42, class_weight='balanced'),
    "SVM": SVC(kernel='linear', probability=False, random_state=42),
    "XGBoost": XGBClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1
    )
}

for name, clf in cv_classifiers.items():
    print(f"\n[INFO] Cross-validating: {name}")
    metrics = defaultdict(list)

    for fold, (train_idx, test_idx) in enumerate(skf.split(train_features, trainY), 1):
        X_train_cv, X_test_cv = train_features[train_idx], train_features[test_idx]
        y_train_cv, y_test_cv = trainY[train_idx], trainY[test_idx]

        clf.fit(X_train_cv, y_train_cv)
        preds = clf.predict(X_test_cv)

        metrics["Accuracy"].append(accuracy_score(y_test_cv, preds))
        metrics["Precision"].append(precision_score(y_test_cv, preds, average='weighted', zero_division=0))
        metrics["Recall"].append(recall_score(y_test_cv, preds, average='weighted', zero_division=0))
        metrics["F1 Score"].append(f1_score(y_test_cv, preds, average='weighted', zero_division=0))

    print(f"{name} - CV Accuracy:  {np.mean(metrics['Accuracy']):.4f}")
    print(f"{name} - CV Precision: {np.mean(metrics['Precision']):.4f}")
    print(f"{name} - CV Recall:    {np.mean(metrics['Recall']):.4f}")
    print(f"{name} - CV F1 Score:  {np.mean(metrics['F1 Score']):.4f}")

# ------------------- Final Evaluation on Validation Set -------------------
print("\n[INFO] Final Training and Evaluation on Validation Set...")

for name, clf in cv_classifiers.items():
    print(f"\n[INFO] Training: {name}")
    clf.fit(train_features, trainY)
    train_preds = clf.predict(train_features)
    val_preds = clf.predict(val_features)

    print(f"\n{name} - Training Metrics")
    print(f"Accuracy:  {accuracy_score(trainY, train_preds):.4f}")
    print(f"Precision: {precision_score(trainY, train_preds, average='weighted'):.4f}")
    print(f"Recall:    {recall_score(trainY, train_preds, average='weighted'):.4f}")
    print(f"F1 Score:  {f1_score(trainY, train_preds, average='weighted'):.4f}")

    print(f"\n{name} - Validation Metrics")
    print(f"Accuracy:  {accuracy_score(valY, val_preds):.4f}")
    print(f"Precision: {precision_score(valY, val_preds, average='weighted'):.4f}")
    print(f"Recall:    {recall_score(valY, val_preds, average='weighted'):.4f}")
    print(f"F1 Score:  {f1_score(valY, val_preds, average='weighted'):.4f}")

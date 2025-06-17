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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from xgboost import XGBClassifier


# ------------------- Preprocessing -------------------
def preprocess_image(image_path, target_size=(300, 300)):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    image = cv2.resize(image, target_size)
    image = image.astype("float32") / 255.0
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
num_classes = len(le.classes_)


# ------------------- Feature Extraction -------------------
print("[INFO] Extracting features using EfficientNetB3 + GAP...")
base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(300, 300, 3))
base_model.trainable = False

from tensorflow.keras.layers import Dropout

x = GlobalAveragePooling2D()(base_model.output)
model = Model(inputs=base_model.input, outputs=x)


train_features = model.predict(trainX, batch_size=16, verbose=1)
val_features = model.predict(valX, batch_size=16, verbose=1)

# ------------------- Feature Normalization -------------------
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
val_features = scaler.transform(val_features)

# ------------------- PCA -------------------
# print("[INFO] Applying PCA to reduce dimensionality...")
# pca = PCA(n_components=256, random_state=42)
# train_features = pca.fit_transform(train_features)
# val_features = pca.transform(val_features)

# ------------------- Cross-Validation -------------------
print("\n[INFO] Performing 5-Fold Stratified Cross-Validation on Training Data...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=5, min_samples_split=10, min_samples_leaf=4, n_jobs=-1, random_state=42, class_weight='balanced'),
    "SVM": SVC(kernel='linear', probability=False, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
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

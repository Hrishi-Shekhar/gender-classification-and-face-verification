import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import pickle
from deepface import DeepFace

# ------------------- Configuration -------------------
SEED = 42
EMBED_DIM = 512
PAIR_CACHE = 'cached_pairs.npz'
MODEL_PATH = 'siamese_model.pth'
PREPROCESSED_DIR = 'preprocessed'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ------------------- Dataset & Model -------------------
class SiameseDataset(Dataset):
    def __init__(self, p1, p2, labels):
        self.p1 = torch.tensor(p1, dtype=torch.float32)
        self.p2 = torch.tensor(p2, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)
      
    def __getitem__(self, idx):
        return self.p1[idx], self.p2[idx], self.labels[idx]

class SiameseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(EMBED_DIM * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, a, b):
        concat = torch.cat([a, b], dim=1)
        return self.fc(concat).squeeze()

# ------------------- Utility functions -------------------
def load_pickle_data(split):
    pkl_path = os.path.join(PREPROCESSED_DIR, f"{split}_data.pkl")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Pickle file for {split} data not found at {pkl_path}")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded {split} data from {pkl_path}")
    return data

def load_cached_pairs():
    if not os.path.exists(PAIR_CACHE):
        raise FileNotFoundError(f"Cached pairs file {PAIR_CACHE} not found.")
    cached = np.load(PAIR_CACHE)
    p1, p2, labels = cached['p1'], cached['p2'], cached['labels']
    print(f"Loaded cached pairs from {PAIR_CACHE}")
    return p1, p2, labels

# ------------------- Training & Evaluation -------------------
def train(model, loader, epochs=10):
    model.train()
    model.to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()
    for epoch in range(epochs):
        total_loss = 0
        for a, b, y in loader:
            a, b, y = a.to(DEVICE), b.to(DEVICE), y.to(DEVICE)
            optim.zero_grad()
            preds = model(a, b)
            loss = loss_fn(preds, y)
            loss.backward()
            optim.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

def evaluate(data, model, split_name="VAL", threshold=0.5):
    model.eval()
    model.to(DEVICE)
    y_true, y_pred = [], []
    with torch.no_grad():
        for person_id, sets in tqdm(data.items(), desc=f"Evaluating [{split_name}]"):
            clean = sets.get('clean', [])
            distorted = sets.get('distorted', [])
            if not clean or not distorted:
                continue
            clean_tensor = torch.tensor(clean, dtype=torch.float32).to(DEVICE)
            for d_emb in distorted:
                d_tensor = torch.tensor(d_emb, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                d_repeated = d_tensor.repeat(len(clean_tensor), 1)
                scores = model(d_repeated, clean_tensor).cpu().numpy()
                pred = person_id if np.max(scores) > threshold else "unknown"
                y_true.append(person_id)
                y_pred.append(pred)
    y_true_binary = [1] * len(y_true)
    y_pred_binary = [1 if p == t else 0 for p, t in zip(y_pred, y_true)]
    acc = accuracy_score(y_true_binary, y_pred_binary)
    precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
    print(f"\n[{split_name}] Accuracy:  {acc:.4f}")
    print(f"[{split_name}] Precision: {precision:.4f}")
    print(f"[{split_name}] Recall:    {recall:.4f}")
    print(f"[{split_name}] F1 Score:  {f1:.4f}")

# ------------------- Test directory evaluation -------------------
def evaluate_unseen_test(test_dir, model, embed_cache_dir='embed_cache', detector_backend='opencv', embed_model='ArcFace'):
    from deepface.commons import functions
    import glob

    if not os.path.exists(test_dir):
        print(f"Test directory {test_dir} does not exist.")
        return

    print(f"\nEvaluating unseen test images from: {test_dir}")

    train_data = load_pickle_data('train')
    val_data = load_pickle_data('val')
    known_embeddings = []
    known_labels = []

    for split_data in [train_data, val_data]:
        for pid, sets in split_data.items():
            for emb in sets.get('clean', []) + sets.get('distorted', []):
                known_embeddings.append(emb)
                known_labels.append(pid)
    known_embeddings = np.array(known_embeddings)

    model.eval()
    model.to(DEVICE)

    test_image_paths = []
    for ext in ('.jpg', '.jpeg', '*.png'):
        test_image_paths.extend(glob.glob(os.path.join(test_dir, ext)))

    y_true = []
    y_pred = []

    for img_path in tqdm(test_image_paths, desc="Processing test images"):
        try:
            faces = DeepFace.extract_faces(img_path=img_path, detector_backend=detector_backend,
                                           enforce_detection=False, align=True)
            if not faces:
                continue
            face_img = faces[0]['face'] / 255.0
            emb = DeepFace.represent(face_img, model_name=embed_model, enforce_detection=False)[0]['embedding']
            emb_tensor = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            known_tensor = torch.tensor(known_embeddings, dtype=torch.float32).to(DEVICE)
            emb_repeated = emb_tensor.repeat(len(known_tensor), 1)
            scores = model(emb_repeated, known_tensor).cpu().numpy()

            max_idx = np.argmax(scores)
            max_score = scores[max_idx]
            pred_label = known_labels[max_idx] if max_score > 0.5 else "unknown"

            print(f"Image: {os.path.basename(img_path)} Predicted: {pred_label} (score={max_score:.3f})")

        except Exception as e:
            print(f"Failed to process {img_path}: {e}")

# ------------------- Main pipeline -------------------
def run_siamese_pipeline(test_dir=None):
    train_data = load_pickle_data('train')
    val_data = load_pickle_data('val')

    p1, p2, labels = load_cached_pairs()
    train_loader = DataLoader(SiameseDataset(p1, p2, labels), batch_size=64, shuffle=True)

    model = SiameseModel()
    if os.path.exists(MODEL_PATH):
        print(f"\nLoading model from {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    else:
        print("Model not found, training...")
        train(model, train_loader, epochs=10)
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")

    evaluate(train_data, model, split_name="TRAIN")
    evaluate(val_data, model, split_name="VAL")

    if test_dir:
        evaluate_unseen_test(test_dir, model)

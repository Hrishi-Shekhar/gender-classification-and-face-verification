# siamese_pipeline.py

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from deepface import DeepFace
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import pickle

# ------------------- Configuration -------------------
SEED = 42
IMAGE_SIZE = (160, 160)
EMBED_DIM = 512
DETECTOR = 'opencv'
EMBED_MODEL = 'ArcFace'
FACE_CACHE = 'face_cache'
EMBED_CACHE = 'embed_cache'
PAIR_CACHE = 'cached_pairs.npz'
MODEL_PATH = 'siamese_model.pth'
PREPROCESSED_DIR = 'preprocessed'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.makedirs(FACE_CACHE, exist_ok=True)
os.makedirs(EMBED_CACHE, exist_ok=True)
os.makedirs(PREPROCESSED_DIR, exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ------------------- Utility Functions -------------------
def get_cached_face(split, person_id, img_path):
    key = f"{split}_{person_id}_{os.path.basename(img_path)}"
    cache_path = os.path.join(FACE_CACHE, key + '.npy')
    if os.path.exists(cache_path):
        return np.load(cache_path)
    try:
        faces = DeepFace.extract_faces(img_path=img_path, detector_backend=DETECTOR,
                                       enforce_detection=False, align=True)
        if not faces:
            return None
        face_img = faces[0]['face'] / 255.0
        np.save(cache_path, face_img)
        return face_img
    except Exception as e:
        print(f"Face extraction failed: {img_path} - {e}")
        return None

def get_cached_embedding(split, person_id, img_path):
    key = f"{split}_{person_id}_{os.path.basename(img_path)}"
    cache_path = os.path.join(EMBED_CACHE, key + '.npy')
    if os.path.exists(cache_path):
        return np.load(cache_path)
    face = get_cached_face(split, person_id, img_path)
    if face is None:
        return None
    try:
        result = DeepFace.represent(face, model_name=EMBED_MODEL, enforce_detection=False)[0]['embedding']
        np.save(cache_path, result)
        return result
    except Exception as e:
        print(f"Embedding failed: {img_path} - {e}")
        return None

def load_embeddings(folder, split='train', max_images_per_person=50):
    pkl_path = os.path.join(PREPROCESSED_DIR, f"{split}_data.pkl")
    if os.path.exists(pkl_path):
        print(f"\nLoaded {split}_data from cache: {pkl_path}")
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)

    data = {}
    for person_id in tqdm(os.listdir(folder), desc=f"Loading {split} embeddings"):
        person_path = os.path.join(folder, person_id)
        if not os.path.isdir(person_path):
            continue
        clean_embs, distorted_embs = [], []
        clean_imgs = [f for f in os.listdir(person_path) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
        for f in clean_imgs[:max_images_per_person]:
            emb = get_cached_embedding(split, person_id, os.path.join(person_path, f))
            if emb is not None:
                clean_embs.append(emb)

        distortion_path = os.path.join(person_path, 'distortion')
        if os.path.exists(distortion_path):
            distorted_imgs = [f for f in os.listdir(distortion_path) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
            for f in distorted_imgs[:max_images_per_person]:
                emb = get_cached_embedding(split, person_id, os.path.join(distortion_path, f))
                if emb is not None:
                    distorted_embs.append(emb)

        if len(clean_embs + distorted_embs) >= 2:
            data[person_id] = {'clean': clean_embs, 'distorted': distorted_embs}

    if split in ['train', 'val']:
        with open(pkl_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"\nSaved {split}_data to: {pkl_path}")
    return data

def create_embedding_pairs(data, max_pairs_per_class=10):
    if os.path.exists(PAIR_CACHE):
        print("\nLoading cached pairs...")
        cached = np.load(PAIR_CACHE)
        return cached['p1'], cached['p2'], cached['labels']

    pairs, labels = [], []
    identities = list(data.keys())
    for person_id in tqdm(identities, desc="Generating pairs"):
        embs = data[person_id]['clean'] + data[person_id]['distorted']
        for _ in range(min(max_pairs_per_class, len(embs))):
            a, b = random.sample(embs, 2)
            pairs.append((a, b))
            labels.append(1)

        for _ in range(min(max_pairs_per_class, len(embs))):
            a = random.choice(embs)
            hardest_neg = None
            max_sim = -1
            for neg_id in identities:
                if neg_id == person_id:
                    continue
                for neg_emb in data[neg_id]['clean'] + data[neg_id]['distorted']:
                    sim = cosine_similarity([a], [neg_emb])[0][0]
                    if sim > max_sim:
                        max_sim = sim
                        hardest_neg = neg_emb
            if hardest_neg is not None:
                pairs.append((a, hardest_neg))
                labels.append(0)

    p1, p2 = zip(*pairs)
    np.savez(PAIR_CACHE, p1=p1, p2=p2, labels=labels)
    return np.array(p1), np.array(p2), np.array(labels)

# ------------------- Dataset & Model -------------------
class SiameseDataset(Dataset):
    def __init__(self, p1, p2, labels):
        self.p1 = torch.tensor(p1, dtype=torch.float32)
        self.p2 = torch.tensor(p2, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    def __len__(self): return len(self.labels)
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

# ------------------- Train & Evaluation -------------------
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

def evaluate_split(data, model, split_name="VAL", threshold=0.5):
    model.eval()
    model.to(DEVICE)
    y_true, y_pred = [], []
    with torch.no_grad():
        for person_id, sets in tqdm(data.items(), desc=f"Evaluating [{split_name}]"):
            clean = sets['clean']
            distorted = sets['distorted']
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

# ------------------- Main Callable Function -------------------
def run_siamese_pipeline(train_dir=None, val_dir=None, test_dir=None):
    if train_dir and val_dir:
        train_data = load_embeddings(train_dir, 'train')
        val_data = load_embeddings(val_dir, 'val')
    else:
        print("Error: Training and validation directories must be provided.")
        return

    test_data = load_embeddings(test_dir, 'test') if test_dir else None

    p1, p2, labels = create_embedding_pairs(train_data)
    train_loader = DataLoader(SiameseDataset(p1, p2, labels), batch_size=64, shuffle=True)

    model = SiameseModel()

    if os.path.exists(MODEL_PATH):
        print(f"\nLoading model from {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    else:
        print("\nTraining model...")
        train(model, train_loader, epochs=10)
        print(f"\nSaving model to {MODEL_PATH}")
        torch.save(model.state_dict(), MODEL_PATH)

    evaluate_split(train_data, model, split_name="TRAIN")
    evaluate_split(val_data, model, split_name="VAL")
    if test_data:
        evaluate_split(test_data, model, split_name="TEST")



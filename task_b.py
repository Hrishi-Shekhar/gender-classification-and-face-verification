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

# --------------------- Configuration ---------------------
SEED = 42
EMBED_DIM = 512
DETECTOR = 'opencv'
EMBED_MODEL = 'ArcFace'
FACE_CACHE = 'face_cache'
EMBED_CACHE = 'embed_cache'
PAIR_CACHE = 'cached_pairs.npz'
MODEL_PATH = 'siamese_model.pth'
PREPROCESSED_DIR = 'preprocessed'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# --------------------- Dataset ---------------------
class SiameseDataset(Dataset):
    def __init__(self, p1, p2, labels):
        self.p1 = torch.tensor(p1, dtype=torch.float32)
        self.p2 = torch.tensor(p2, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return self.p1[idx], self.p2[idx], self.labels[idx]

# --------------------- Model ---------------------
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

# --------------------- Load Preprocessed Data ---------------------
def load_preprocessed_data():
    train_pkl = os.path.join(PREPROCESSED_DIR, "train_data.pkl")
    val_pkl = os.path.join(PREPROCESSED_DIR, "val_data.pkl")
    if not (os.path.exists(train_pkl) and os.path.exists(val_pkl)):
        raise FileNotFoundError("Preprocessed train or val data not found. Please preprocess before testing.")
    with open(train_pkl, 'rb') as f:
        train_data = pickle.load(f)
    with open(val_pkl, 'rb') as f:
        val_data = pickle.load(f)
    return train_data, val_data

# --------------------- Create Embedding Pairs ---------------------
def create_embedding_pairs(data, max_pairs_per_class=10):
    if not os.path.exists(PAIR_CACHE):
        raise FileNotFoundError("Cached pairs file not found. Please generate pairs before testing.")
    cached = np.load(PAIR_CACHE)
    return cached['p1'], cached['p2'], cached['labels']

# --------------------- Evaluation ---------------------
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

# --------------------- Main ---------------------
def main(test_dir):
    if test_dir is None or not os.path.exists(test_dir):
        raise ValueError("Please provide a valid test directory path")

    # Load preprocessed data (train and val)
    train_data, val_data = load_preprocessed_data()

    # Load cached pairs from train data (needed to match input size)
    p1, p2, labels = create_embedding_pairs(train_data)
    test_data = None

    # Load test data embeddings on-the-fly (since test data is new)
    from task_b import load_embeddings  # Assuming load_embeddings is implemented exactly as in your code
    test_data = load_embeddings(test_dir, split='test')

    # Load model and weights
    model = SiameseModel()
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model weights not found at {MODEL_PATH}")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    # Evaluate all splits
    evaluate_split(train_data, model, split_name="TRAIN")
    evaluate_split(val_data, model, split_name="VAL")
    if test_data:
        evaluate_split(test_data, model, split_name="TEST")

# --------------------- Entry ---------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python test.py <path_to_test_folder>")
        exit(1)
    test_dir = sys.argv[1]
    main(test_dir)

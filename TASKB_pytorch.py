import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from deepface import DeepFace
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import pathlib

MODEL_PATH = 'TASKB_pytorch.pth'

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Config
IMAGE_SIZE = (160, 160)
EMBED_DIM = 512
DETECTOR = 'opencv'
EMBED_MODEL = 'ArcFace'
FACE_CACHE = 'face_cache'
EMBED_CACHE = 'embed_cache'
PAIR_CACHE = 'cached_pairs.npz'
os.makedirs(FACE_CACHE, exist_ok=True)
os.makedirs(EMBED_CACHE, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def l2_normalize(x, axis=-1, eps=1e-10):
    return x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), eps))

#using DeepFace to extract faces and embeddings
# This function caches the extracted faces and embeddings to avoid redundant computations
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

# This function retrieves the cached embedding for a given image, or computes it if not cached
# It uses DeepFace to compute the embedding and caches it for future use
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
        result = l2_normalize(np.array(result))
        np.save(cache_path, result)
        return result
    except Exception as e:
        print(f"Embedding failed: {img_path} - {e}")
        return None

# This function loads embeddings from a specified folder structure
# It expects each person's images to be in a subfolder named after their ID
def load_embeddings(folder, split='train', max_images_per_person=50):
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
    return data

# This function creates pairs of embeddings for training a Siamese network
# It generates positive pairs (same person) and negative pairs (different persons)
def create_embedding_pairs(data, max_pairs_per_class=20):
    if os.path.exists(PAIR_CACHE):
        print("\n✅ Loading cached pairs...")
        cached = np.load(PAIR_CACHE)
        return cached['p1'], cached['p2'], cached['labels']
    pairs, labels = [], []
    identities = list(data.keys())
    for person_id in tqdm(identities, desc="Pair generation"):
        embs = data[person_id]['clean'] + data[person_id]['distorted']
        if len(embs) < 2:
            continue
        for _ in range(min(max_pairs_per_class, len(embs))):
            a, b = random.sample(embs, 2)
            pairs.append((a, b))
            labels.append(1)
        for _ in range(min(max_pairs_per_class, len(embs))):
            a = random.choice(embs)
            neg_embs = [
                emb for neg_id in identities if neg_id != person_id
                for emb in data[neg_id]['clean'] + data[neg_id]['distorted']
            ]
            if not neg_embs:
                continue
            sims = cosine_similarity([a], neg_embs)[0]
            hardest_neg = neg_embs[np.argmax(sims)]
            pairs.append((a, hardest_neg))
            labels.append(0)
    p1, p2 = zip(*pairs)
    np.savez(PAIR_CACHE, p1=p1, p2=p2, labels=labels)
    return np.array(p1), np.array(p2), np.array(labels)

# This class defines a custom dataset for the Siamese network
# It takes pairs of embeddings and their labels as input
class SiameseDataset(Dataset):
    def __init__(self, p1, p2, labels):
        self.p1 = torch.tensor(p1, dtype=torch.float32)
        self.p2 = torch.tensor(p2, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.p1[idx], self.p2[idx], self.labels[idx]

# This class defines the Siamese network model
# It consists of an encoder to process the embeddings and a classifier to output similarity scores
class SiameseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(EMBED_DIM, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()

        )
    def forward(self, a, b):
        a = self.encoder(a)
        b = self.encoder(b)
        diff = torch.abs(a - b)
        diff = nn.functional.normalize(diff, p=2, dim=1)
        diff = nn.functional.dropout(diff, p=0.3, training=self.training)
        diff = nn.functional.relu(diff)
        diff = nn.functional.normalize(diff, p=2, dim=1)
        return self.classifier(diff).squeeze()

# This function trains the Siamese model using the provided dataloader
# It uses AdamW optimizer and binary cross-entropy loss
def train(model, dataloader, epochs=20):
    model.train()
    model.to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    loss_fn = nn.BCELoss()
    for epoch in range(epochs):
        total_loss = 0
        for a, b, y in dataloader:
            a, b, y = a.to(DEVICE), b.to(DEVICE), y.to(DEVICE)
            optim.zero_grad()
            preds = model(a, b)
            loss = loss_fn(preds, y)
            loss.backward()
            optim.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

# This function evaluates the model on a validation or test dataset 
# It computes accuracy and F1 score based on the predictions
# It uses a threshold to determine if the prediction is a known person or "unknown"
def evaluate_on_val(val_data, model, threshold=0.5):
    y_true, y_pred = [], []
    model.eval()
    model.to(DEVICE)
    with torch.no_grad():
        for person_id, sets in tqdm(val_data.items(), desc="Evaluating"):
            clean = sets['clean']
            distorted = sets['distorted']
            if not clean or not distorted:
                continue
            clean = torch.tensor(clean, dtype=torch.float32).to(DEVICE)
            for d in distorted:
                d = torch.tensor(d, dtype=torch.float32).unsqueeze(0).repeat(len(clean), 1).to(DEVICE)
                scores = model(d, clean).cpu().numpy()
                pred = person_id if np.max(scores) > threshold else "unknown"
                y_true.append(person_id)
                y_pred.append(pred)
    print("[VAL] Accuracy:", accuracy_score(y_true, y_pred))
    print("[VAL] F1:", f1_score(y_true, y_pred, average='macro', zero_division=0))

# This is the main function that orchestrates the loading of data, training, and evaluation
# It loads training and validation data, creates pairs, trains the model, and evaluates it
def main(test_dir=None):
    root = r'C:\\Users\\roytu\\Downloads\\TASKB_COMYS\\Comys_Hackathon5 (1)\\Comys_Hackathon5\\Task_B'
    train_data = load_embeddings(os.path.join(root, 'train'), 'train')
    val_data = load_embeddings(os.path.join(root, 'val'), 'val')
    test_data = load_embeddings(test_dir, split='test') if test_dir and os.path.exists(test_dir) else None

    p1, p2, labels = create_embedding_pairs(train_data)
    print(f"Training Pairs: Positive={np.sum(labels==1)}, Negative={np.sum(labels==0)}")
    train_loader = DataLoader(SiameseDataset(p1, p2, labels), batch_size=64, shuffle=True)

    model = SiameseModel()

    if os.path.exists(MODEL_PATH):
        print(f"\n✅ Loading saved model from {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    else:
        print("\n🚀 Training model...")
        train(model, train_loader)
        print(f"\n💾 Saving model to {MODEL_PATH}")
        torch.save(model.state_dict(), MODEL_PATH)

    # Evaluate
    evaluate_on_val(val_data, model)
    if test_data:
        print("\n✅ Evaluating on test set...")
        evaluate_on_val(test_data, model)

 

if __name__ == "__main__":
    main()

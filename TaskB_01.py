import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from scipy.spatial.distance import cosine
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

# Configuration
device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE  = 16
THRESHOLD   = 0.5
EMBEDDING_DIM = 512

# Initialize MTCNN & base FaceNet (in eval mode)
mtcnn = MTCNN(image_size=160, margin=20, min_face_size=20,
              thresholds=[0.6,0.7,0.7], factor=0.709,
              post_process=True, device=device)
base_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

class FaceVerificationModel(nn.Module):
    def __init__(self, base_model, embedding_dim=512):
        super().__init__()
        self.base_model   = base_model
        self.fc1          = nn.Linear(512, 256)
        self.dropout      = nn.Dropout(0.3)
        self.fc2          = nn.Linear(256, embedding_dim)
        self.freeze_backbone = True  # Set to True to freeze the backbone
    
    def forward(self, x):
        if self.freeze_backbone:
            with torch.no_grad():
                feat = self.base_model(x)
        else:
            feat = self.base_model(x)
        x = F.relu(self.fc1(feat))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.normalize(x, p=2, dim=1)

class FaceDataset(Dataset):
    # ... identical to yours ...
    def __init__(self, root_dir, mode='train'):
        # same as before
        self.image_paths = []
        self.identities = []
        self.identity_to_idx = {}
        for idx, identity in enumerate(os.listdir(root_dir)):
            p = os.path.join(root_dir, identity)
            if not os.path.isdir(p): continue
            self.identity_to_idx[identity] = idx
            clean = [f for f in os.listdir(p) if f.lower().endswith(('jpg','png','jpeg'))]
            dist_dir = os.path.join(p, 'distortion')
            distorted = os.listdir(dist_dir) if os.path.exists(dist_dir) else []
            for fn in clean:
                self.image_paths.append(os.path.join(p, fn))
                self.identities.append(idx)
            for fn in distorted:
                self.image_paths.append(os.path.join(dist_dir, fn))
                self.identities.append(idx)

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        idc  = self.identities[idx]
        try:
            img = Image.open(path).convert('RGB')
            face = mtcnn(img)
            if face is None:
                face = torch.zeros(3,160,160)
        except:
            face = torch.zeros(3,160,160)
        return face, idc, path

def process_image(img_path, model):
    """Always run model in eval + no_grad for single-sample embedding."""
    try:
        img = Image.open(img_path).convert('RGB')
        face = mtcnn(img)
        if face is None:
            return None
        model.eval()                     # ensure eval mode
        with torch.no_grad():            # disable grads
            emb = model(face.unsqueeze(0).to(device))
        return emb.cpu().numpy().squeeze()
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

def train_model(model, loader, num_epochs=10, checkpoint_dir='checkpoints', start_epoch=0):
    os.makedirs('checkpoints', exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.007)
    criterion = nn.TripletMarginLoss(margin=0.3)
    model.train()
    for epoch in range(start_epoch, num_epochs):
        # No unfreezing logic, always keep backbone frozen
        total_loss = 0.0
        batches    = 0
        for faces, ids, _ in tqdm(loader, desc=f"Epoch {epoch+1}"):
            if faces.size(0)<2:
                continue
            faces = faces.to(device)
            ids   = ids.to(device)
            if len(faces) < 3: continue
            embeddings = model(faces)
            loss = 0.0
            count = 0
            for i in range(len(embeddings)):
                anc_id = ids[i]
                anchor = embeddings[i:i+1]
                pos_idx = (ids == anc_id).nonzero(as_tuple=True)[0]
                neg_idx = (ids != anc_id).nonzero(as_tuple=True)[0]
                if len(pos_idx) > 1 and len(neg_idx) > 0:
            # Exclude anchor itself from positives
                    pos_idx_wo_anchor = pos_idx[pos_idx != i]
                    if len(pos_idx_wo_anchor) == 0:
                        continue
                    pos_dists = torch.norm(anchor - embeddings[pos_idx_wo_anchor], dim=1)
                    hardest_pos_idx = pos_idx_wo_anchor[torch.argmax(pos_dists)]
                    neg_dists = torch.norm(anchor - embeddings[neg_idx], dim=1)
                    hardest_neg_idx = neg_idx[torch.argmin(neg_dists)]
                    loss += criterion(anchor,
                              embeddings[hardest_pos_idx:hardest_pos_idx+1],
                              embeddings[hardest_neg_idx:hardest_neg_idx+1])
                    count += 1
            if count>0:
                loss = loss / count
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                batches    += 1
        if batches>0:
            print(f"Epoch {epoch+1} avg loss: {total_loss/batches:.4f}")
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        torch.save(checkpoint, os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth"))
        print(f"Checkpoint saved for epoch {epoch+1}")    

def build_reference_database(ref_dir, model):
    model.eval()       # <- set eval once here
    db = {}
    for identity in os.listdir(ref_dir):
        path_id = os.path.join(ref_dir, identity)
        if not os.path.isdir(path_id): continue
        embs = []
        for fn in os.listdir(path_id):
            if fn.lower().endswith(('jpg','png','jpeg')):
                e = process_image(os.path.join(path_id, fn), model)
                if e is not None:
                    embs.append(e)
        if embs:
            db[identity] = np.vstack(embs)
    return db

def verify_against_database(test_path, db, model, threshold=0.5):
    test_emb = process_image(test_path, model)
    if test_emb is None:
        return False, 0.0, None
    best_sim, best_id = 0.0, None
    for identity, ems in db.items():
        sims = 1 - np.linalg.norm(ems - test_emb, axis=1)  # since both L2-normalized
        m = sims.max()
        if m > best_sim:
            best_sim, best_id = m, identity
    return (best_sim>threshold), best_sim, best_id

def evaluate_verification_system(val_dir, db, model, threshold=0.5):
    results = []
    for identity in tqdm(os.listdir(val_dir), desc="Evaluating"):
        p_id = os.path.join(val_dir, identity, 'distortion')
        if not os.path.isdir(p_id): continue
        for fn in os.listdir(p_id):
            if not fn.lower().endswith(('jpg','png','jpeg')): continue
            tp = os.path.join(p_id, fn)
            match, sim, pred = verify_against_database(tp, db, model, threshold)
            correct = (match and pred==identity)
            results.append({
                'test_image': tp,
                'true_identity': identity,
                'predicted_identity': pred,
                'is_match': match,
                'similarity': sim,
                'correct': correct
            })
    return results

def calculate_metrics(results):
    sims = [r['similarity'] for r in results]
    labels = [1 if r['correct'] else 0 for r in results]
    acc = sum(r['correct'] for r in results) / len(results)
    auc = roc_auc_score(labels, sims) if len(set(labels))>1 else 0.0
    print(f"Total: {len(results)}, Accuracy: {acc:.4f}, AUC: {auc:.4f}")

if __name__ == '__main__':
    DATA_ROOT = r'C:\Users\roytu\Downloads\TASKB_COMYS\Comys_Hackathon5 (1)\Comys_Hackathon5\Task_B'
    model = FaceVerificationModel(base_model, EMBEDDING_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.007)
    # ---- Load checkpoint if resuming ----
    checkpoint_path = "checkpoints/checkpoint_epoch_5.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming from epoch {start_epoch}")
        # Always keep backbone frozen after checkpoint resume
        model.freeze_backbone = True
        for param in model.base_model.parameters():
            param.requires_grad = False
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.007
        print("Backbone frozen and learning rate set to 0.007 after checkpoint resume.")
    else:
        start_epoch = 0
    
    
    
    
    # 1) Train (optional)
    train_ds = FaceDataset(os.path.join(DATA_ROOT, 'train'))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    train_model(model, train_loader, num_epochs=10)

    # 2) Switch to eval for all subsequent steps
    model.eval()

    # 3) Build gallery of clean embeddings
    reference_db = build_reference_database(os.path.join(DATA_ROOT, 'val'),model)

    # 4) Run on distorted validation images
    results = evaluate_verification_system(
        os.path.join(DATA_ROOT, 'val'),
        reference_db,
        model,
        threshold=THRESHOLD
    )

    # 5) Compute metrics
    calculate_metrics(results)

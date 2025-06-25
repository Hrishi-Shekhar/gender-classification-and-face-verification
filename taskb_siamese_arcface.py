import os
import random
import numpy as np
import tensorflow as tf
from deepface import DeepFace
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import pickle

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

def create_embedding_pairs(data, max_pairs_per_class=5):
    if os.path.exists(PAIR_CACHE):
        print("\n✅ Loading cached pairs...")
        cached = np.load(PAIR_CACHE)
        return cached['p1'], cached['p2'], cached['labels']

    pairs, labels = [], []
    identities = list(data.keys())

    for person_id in tqdm(identities, desc="Pair generation"):
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
    p1, p2, labels = np.array(p1), np.array(p2), np.array(labels)
    np.savez(PAIR_CACHE, p1=p1, p2=p2, labels=labels)
    return p1, p2, labels

def build_siamese_model(input_dim=EMBED_DIM):
    input_a = tf.keras.Input(shape=(input_dim,))
    input_b = tf.keras.Input(shape=(input_dim,))
    l1 = tf.keras.layers.Lambda(lambda x: tf.abs(x[0] - x[1]))([input_a, input_b])
    out = tf.keras.layers.Dense(1, activation='sigmoid')(l1)
    model = tf.keras.Model([input_a, input_b], out)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def evaluate_on_val(val_data, model, threshold=0.5):
    y_true, y_pred = [], []
    for person_id, sets in tqdm(val_data.items(), desc="Evaluating"):
        clean_embs = sets['clean']
        distorted_embs = sets['distorted']
        if not clean_embs or not distorted_embs:
            continue
        c_stack = np.stack(clean_embs)
        for d_emb in distorted_embs:
            d_stack = np.repeat(np.expand_dims(d_emb, 0), len(c_stack), axis=0)
            sims = model.predict([d_stack, c_stack], verbose=0).flatten()
            pred = person_id if np.max(sims) > threshold else "unknown"
            y_true.append(person_id)
            y_pred.append(pred)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    print(f"\n[VAL] Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")

def main():
    root = 'Comys_Hackathon5 (1)/Comys_Hackathon5/Task_B'
    train_root = os.path.join(root, 'train')
    val_root = os.path.join(root, 'val')

    print("\n📥 Loading embeddings...")
    train_data = load_embeddings(train_root, split='train')
    val_data = load_embeddings(val_root, split='val')

    print("\n🔍 Creating pairs with hard negatives...")
    p1, p2, labels = create_embedding_pairs(train_data)

    print("\n🧠 Building Siamese model...")
    model = build_siamese_model()

    print("\n🚀 Training...")
    model.fit([p1, p2], labels, batch_size=32, epochs=10, validation_split=0.1)

    print("\n✅ Evaluating on validation set...")
    evaluate_on_val(val_data, model)

if __name__ == "__main__":
    main()

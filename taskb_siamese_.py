import os
import random
import numpy as np
import tensorflow as tf
from deepface import DeepFace
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import cv2
import time

# Config
DETECTOR = 'opencv'  # Options: 'retinaface', 'mtcnn', 'opencv'
CACHE_DIR = 'face_cache'
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cached_face(person_id, img_path, image_size=(160, 160)):
    cache_path = os.path.join(CACHE_DIR, f"{person_id}_{os.path.basename(img_path)}.npy")
    if os.path.exists(cache_path):
        return np.load(cache_path)

    try:
        faces = DeepFace.extract_faces(img_path=img_path, detector_backend=DETECTOR,
                                       enforce_detection=False, align=True)
        if not faces:
            return None
        face_img = cv2.resize(np.array(faces[0]['face']) / 255.0, image_size)
        np.save(cache_path, face_img)
        return face_img
    except Exception as e:
        print(f"Face extraction failed: {img_path} - {e}")
        return None

def load_faces(folder, image_size=(160, 160), max_images_per_person=50):
    data = {}
    start_time = time.time()
    for person_id in tqdm(os.listdir(folder), desc="Loading training faces"):
        person_path = os.path.join(folder, person_id)
        if not os.path.isdir(person_path):
            continue

        faces = []

        clean_imgs = [f for f in os.listdir(person_path) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
        for f in clean_imgs[:max_images_per_person]:
            img_path = os.path.join(person_path, f)
            face = get_cached_face(person_id, img_path, image_size)
            if face is not None and face.shape[:2] == image_size:
                faces.append(face)

        distortion_path = os.path.join(person_path, 'distortion')
        if os.path.exists(distortion_path):
            distorted_imgs = [f for f in os.listdir(distortion_path) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
            for f in distorted_imgs[:max_images_per_person]:
                img_path = os.path.join(distortion_path, f)
                face = get_cached_face(person_id, img_path, image_size)
                if face is not None and face.shape[:2] == image_size:
                    faces.append(face)

        if len(faces) >= 2:
            data[person_id] = faces
    print(f"\nFinished loading faces in {time.time() - start_time:.2f} seconds.")
    return data

def create_pairs(data, image_size=(160, 160), max_pairs_per_class=10):
    pairs, labels = [], []
    identities = list(data.keys())

    for person_id in tqdm(identities, desc="Creating pairs"):
        faces = data[person_id]
        if len(faces) < 2:
            continue

        for _ in range(min(max_pairs_per_class, len(faces))):
            f1, f2 = random.sample(faces, 2)
            pairs.append((f1, f2))
            labels.append(1)

        for _ in range(min(max_pairs_per_class, len(faces))):
            neg_id = random.choice([pid for pid in identities if pid != person_id])
            f1 = random.choice(faces)
            f2 = random.choice(data[neg_id])
            pairs.append((f1, f2))
            labels.append(0)

    p1, p2 = zip(*pairs)
    return np.array(p1, dtype=np.float32), np.array(p2, dtype=np.float32), np.array(labels, dtype=np.float32)

def build_siamese_model(input_shape):
    def build_base_cnn():
        inputs = tf.keras.Input(shape=input_shape)
        x = tf.keras.layers.Conv2D(64, (7,7), activation='relu', padding='same')(inputs)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Conv2D(128, (5,5), activation='relu', padding='same')(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same')(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)
        return tf.keras.Model(inputs, x)

    base_cnn = build_base_cnn()

    input_a = tf.keras.Input(shape=input_shape)
    input_b = tf.keras.Input(shape=input_shape)

    encoded_a = base_cnn(input_a)
    encoded_b = base_cnn(input_b)

    l1_distance = tf.keras.layers.Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))([encoded_a, encoded_b])
    output = tf.keras.layers.Dense(1, activation='sigmoid')(l1_distance)

    siamese_model = tf.keras.Model([input_a, input_b], output)
    siamese_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return siamese_model, base_cnn

def evaluate_model_on_val(val_root, embedder, threshold=0.5, image_size=(160, 160)):
    y_true, y_pred = [], []
    start_time = time.time()

    for person_id in tqdm(os.listdir(val_root), desc="Evaluating validation set"):
        person_path = os.path.join(val_root, person_id)
        if not os.path.isdir(person_path):
            continue

        clean_imgs = [os.path.join(person_path, f) for f in os.listdir(person_path)
                      if os.path.isfile(os.path.join(person_path, f)) and f.lower().endswith(('jpg', 'jpeg', 'png'))]

        distortion_path = os.path.join(person_path, 'distortion')
        if not os.path.exists(distortion_path):
            continue

        clean_faces = [get_cached_face(person_id, p, image_size) for p in clean_imgs]
        clean_faces = [f for f in clean_faces if f is not None and f.shape[:2] == image_size]
        if len(clean_faces) == 0:
            continue

        for fname in os.listdir(distortion_path):
            img_path = os.path.join(distortion_path, fname)
            face = get_cached_face(person_id, img_path, image_size)
            if face is None or face.shape[:2] != image_size:
                continue

            f_emb = embedder.predict(np.expand_dims(face, axis=0), verbose=0)[0]
            sims = []
            for c in clean_faces:
                c_emb = embedder.predict(np.expand_dims(c, axis=0), verbose=0)[0]
                sim = cosine_similarity([f_emb], [c_emb])[0][0]
                sims.append(sim)

            best_sim = np.max(sims)
            pred = person_id if best_sim > threshold else "unknown"

            y_true.append(person_id)
            y_pred.append(pred)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    print(f"\n[VAL SET] Top-1 Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
    print(f"Validation completed in {time.time() - start_time:.2f} seconds.")

def main():
    root = 'Comys_Hackathon5 (1)/Comys_Hackathon5/Task_B'
    train_root = os.path.join(root, 'train')
    val_root = os.path.join(root, 'val')

    assert set(os.listdir(train_root)).isdisjoint(os.listdir(val_root)), "Train/Val overlap detected!"

    print("\n📥 Loading training faces...")
    data = load_faces(train_root, image_size=(160, 160))

    print("\n🧩 Creating training pairs...")
    p1, p2, labels = create_pairs(data, image_size=(160, 160), max_pairs_per_class=10)

    print("\n🧠 Building Siamese model...")
    model, embedder = build_siamese_model((160, 160, 3))

    print("\n🚀 Training model...")
    train_start = time.time()
    model.fit([p1, p2], labels, batch_size=32, epochs=10, validation_split=0.1)
    print(f"Training completed in {time.time() - train_start:.2f} seconds.")

    print("\n🔍 Evaluating on disjoint validation set...")
    evaluate_model_on_val(val_root, embedder, threshold=0.5)

if __name__ == "__main__":
    main()
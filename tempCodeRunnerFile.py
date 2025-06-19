print("[INFO] Applying PCA to reduce dimensionality...")
pca = PCA(n_components=128, random_state=42)
train_features = pca.fit_transform(train_features)
val_features = pca.transform(val_features)
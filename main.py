import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -----------------------------
# Load Dataset
# -----------------------------
DATASET_PATH = "dataset"
IMAGE_SIZE = (100, 100)

X = []
y = []
label_map = {}
label = 0

for person_name in os.listdir(DATASET_PATH):
    person_path = os.path.join(DATASET_PATH, person_name)
    if not os.path.isdir(person_path):
        continue

    label_map[label] = person_name

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, IMAGE_SIZE)
        X.append(img.flatten())
        y.append(label)

    label += 1

X = np.array(X)
y = np.array(y)

print("Total images:", X.shape[0])
print("Image vector size:", X.shape[1])

# -----------------------------
# Train-Test Split (60%-40%)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)

# -----------------------------
# PCA + ANN for different k
# -----------------------------
max_k = min(X_train.shape[0] - 1, X_train.shape[1])

k_values = [k for k in [5, 10, 20, 30, 40, 50] if k <= max_k]

if not k_values:
    k_values = [max_k]

accuracies = []

for k in k_values:
    print(f"\nTraining with k = {k}")

    pca = PCA(n_components=k, whiten=True)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    ann = MLPClassifier(
        hidden_layer_sizes=(100,),
        max_iter=500,
        random_state=42
    )

    ann.fit(X_train_pca, y_train)

    y_pred = ann.predict(X_test_pca)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

    print(f"Accuracy for k={k}: {acc:.2f}")

# -----------------------------
# Plot Accuracy vs k
# -----------------------------
plt.plot(k_values, accuracies, marker='o')
plt.xlabel("Number of Principal Components (k)")
plt.ylabel("Accuracy")
plt.title("Accuracy vs k (PCA + ANN)")
plt.grid()
plt.show()

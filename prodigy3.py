import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import warnings

# Function to load and preprocess images
def load_images(folder_path):
    images = []
    labels = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        if img_path.endswith(".jpg") or img_path.endswith(".png"):
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to read image: {img_path}")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            img = cv2.resize(img, (100, 100))  # Resize image to a fixed size
            images.append(img.flatten())  # Flatten image matrix
            label = 1 if "dog" in filename else 0  # Assign label based on filename
            labels.append(label)
    return np.array(images), np.array(labels)

# Load images and labels
folder_path = r"C:\Users\vigne\OneDrive\Desktop\PRODIGY\MACHINE LEARNING\task03\images"
images, labels = load_images(folder_path)

if len(images) == 0:
    print("No valid images found in the specified folder.")
    exit()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA for dimensionality reduction
n_components = min(X_train_scaled.shape[0], X_train_scaled.shape[1])
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Train SVM model
svm = SVC(kernel='rbf', C=1.0, gamma='scale')
svm.fit(X_train_pca, y_train)

# Predictions
y_pred = svm.predict(X_test_pca)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred, zero_division=1))

# 2D visualization of decision boundary
def plot_decision_boundary(X, y, model):
    # Create a mesh to plot the decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=plt.cm.coolwarm)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(f'SVM Decision Boundary with PCA-reduced Data (Accuracy: {accuracy:.2f})')
    plt.show()

# Reduce data to 2 dimensions
pca_2d = PCA(n_components=2)
X_train_pca_2d = pca_2d.fit_transform(X_train_scaled)
X_test_pca_2d = pca_2d.transform(X_test_scaled)

# Retrain the SVM on 2D PCA data
svm_2d = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_2d.fit(X_train_pca_2d, y_train)

# Plot decision boundary
plot_decision_boundary(X_train_pca_2d, y_train, svm_2d)

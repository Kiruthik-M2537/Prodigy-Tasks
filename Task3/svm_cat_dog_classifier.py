import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

cat_path = "PetImages/Cat"
dog_path = "PetImages/Dog"

images = []
labels = []

print("Loading Cat Images...")

for file in os.listdir(cat_path)[:500]:
    path = os.path.join(cat_path, file)

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is not None:
        img = cv2.resize(img, (64, 64))
        images.append(img.flatten())
        labels.append(0)

print("Loading Dog Images...")

for file in os.listdir(dog_path)[:500]:
    path = os.path.join(dog_path, file)

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is not None:
        img = cv2.resize(img, (64, 64))
        images.append(img.flatten())
        labels.append(1)

X = np.array(images)
y = np.array(labels)

print("Dataset Shape:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training SVM...")

model = SVC(kernel="linear")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(6,5))
plt.imshow(cm)
plt.colorbar()
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
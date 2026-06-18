import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay

DATASET_PATH = "leapGestRecog"

images = []
labels = []

print("Loading Dataset...")

for subject in os.listdir(DATASET_PATH):

    subject_path = os.path.join(DATASET_PATH, subject)

    if not os.path.isdir(subject_path):
        continue

    for gesture in os.listdir(subject_path):

        gesture_path = os.path.join(subject_path, gesture)

        if not os.path.isdir(gesture_path):
            continue

        count = 0

        for image_file in os.listdir(gesture_path):

            if count >= 100:
                break

            image_path = os.path.join(
                gesture_path,
                image_file
            )

            img = cv2.imread(
                image_path,
                cv2.IMREAD_GRAYSCALE
            )

            if img is None:
                continue

            img = cv2.resize(img, (64, 64))

            images.append(img.flatten())
            labels.append(gesture)

            count += 1

X = np.array(images)
y = np.array(labels)

print("Dataset Shape:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training SVM Model...")

model = SVC(kernel="linear")

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print(
    "\nAccuracy:",
    round(accuracy * 100, 2),
    "%"
)

print(
    "\nClassification Report:\n"
)

print(
    classification_report(
        y_test,
        predictions
    )
)

ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions,
    xticks_rotation=90
)

plt.title(
    "Hand Gesture Recognition Confusion Matrix"
)

plt.tight_layout()

plt.show()
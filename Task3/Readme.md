# Cats vs Dogs Image Classification using Support Vector Machine (SVM)

## Overview

This project implements a Support Vector Machine (SVM) classifier to distinguish between images of cats and dogs. The images are preprocessed by converting them to grayscale, resizing them, and transforming them into numerical feature vectors suitable for machine learning.

## Objective

To build an image classification model capable of identifying whether an image contains a cat or a dog using the Support Vector Machine (SVM) algorithm.

## Dataset

Dataset: Cats and Dogs Image Dataset

Directory Structure:

```text
PetImages
├── Cat
│   ├── 0.jpg
│   ├── 1.jpg
│   └── ...
│
└── Dog
    ├── 0.jpg
    ├── 1.jpg
    └── ...
```

## Technologies Used

- Python
- NumPy
- OpenCV
- Matplotlib
- Scikit-Learn

## Methodology

### Data Preprocessing

- Loaded cat and dog images from the dataset.
- Converted images to grayscale.
- Resized images to 64 × 64 pixels.
- Flattened images into one-dimensional feature vectors.
- Assigned labels:
  - 0 → Cat
  - 1 → Dog

### Model Training

- Split dataset into training and testing sets using an 80:20 ratio.
- Trained a Support Vector Machine (SVM) classifier with a linear kernel.
- Evaluated model performance on unseen test data.

## Evaluation Metrics

The model was evaluated using:

- Accuracy Score
- Confusion Matrix
- Classification Report

## Results

The SVM classifier successfully learned to differentiate between cats and dogs from image features and produced reliable predictions on the test dataset.

## Visualizations

### Confusion Matrix

The confusion matrix illustrates the number of correct and incorrect predictions made by the classifier for both classes.

## Project Structure

```text
PRODIGY_DS_03
│
├── PetImages/
│   ├── Cat/
│   └── Dog/
│
├── svm_cat_dog_classifier.py
├── README.md
│
└── outputs/
    └── confusion_matrix.png
```

## Installation

Install the required libraries:

```bash
pip install numpy matplotlib scikit-learn opencv-python
```

## Usage

Run the classifier:

```bash
python svm_cat_dog_classifier.py
```

## Key Learnings

- Image preprocessing using OpenCV
- Feature extraction from images
- Binary image classification
- Support Vector Machine implementation
- Model evaluation using confusion matrices and accuracy metrics

## Conclusion

This project demonstrates how traditional machine learning techniques such as Support Vector Machines can be applied to image classification problems. Through image preprocessing, feature extraction, and model training, the classifier successfully distinguishes between cat and dog images and provides a foundation for more advanced computer vision applications.
# Hand Gesture Recognition using Support Vector Machine

## Overview

This project implements a Hand Gesture Recognition system using the LeapGestRecog dataset and a Support Vector Machine (SVM) classifier. The model identifies and classifies ten different hand gestures from images.

## Dataset

LeapGestRecog Dataset

Gesture Classes:

- Palm
- L
- Fist
- Fist Moved
- Thumb
- Index
- OK
- Palm Moved
- C
- Down

## Technologies Used

- Python
- OpenCV
- NumPy
- Scikit-Learn
- Matplotlib

## Methodology

1. Loaded gesture images from the dataset.
2. Converted images to grayscale.
3. Resized images to 64×64 pixels.
4. Flattened image data into feature vectors.
5. Trained an SVM classifier.
6. Evaluated performance using accuracy and confusion matrix.

## Results

- Model: Support Vector Machine (SVM)
- Number of Classes: 10
- Image Size: 64×64
- Accuracy: Generated during execution

## Project Structure

```text
Task4/
│
├── leapGestRecog/
├── gesture_recognition.py
├── README.md
└── outputs/
```

## How to Run

```bash
pip install numpy opencv-python scikit-learn matplotlib
python gesture_recognition.py
```

## Conclusion

The SVM model successfully classifies hand gestures from image data and demonstrates the application of machine learning techniques in gesture-based human-computer interaction systems.
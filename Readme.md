# House Price Prediction using Linear Regression

## Overview

This project implements a Linear Regression model to predict house prices based on key housing features such as living area, number of bedrooms, and number of bathrooms. The model is trained and evaluated using the House Prices dataset from Kaggle.

## Objective

To build a machine learning model that predicts the selling price of a house using:

- Ground Living Area (GrLivArea)
- Number of Bedrooms (BedroomAbvGr)
- Number of Full Bathrooms (FullBath)

## Dataset

Dataset: House Prices - Advanced Regression Techniques

Source:
https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data

File Used:
- train.csv

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn

## Methodology

### Data Preparation

- Loaded the dataset using Pandas
- Selected relevant features
- Split data into training and testing sets (80:20 ratio)

### Model Building

- Applied Linear Regression using Scikit-Learn
- Trained the model on the training dataset
- Predicted house prices on the test dataset

### Evaluation Metrics

The model was evaluated using:

- R² Score
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

## Results

| Metric | Value |
|----------|----------|
| R² Score | 0.634 |
| MAE | 35,788 |
| RMSE | 52,976 |

## Visualizations

### 1. Correlation Heatmap

Shows the relationship between selected features and house prices.

### 2. Actual vs Predicted Prices

Compares the model's predictions against actual house prices.

### 3. Feature Importance

Displays the impact of each feature on house price prediction.

## Key Insights

- GrLivArea (living area) has the strongest positive correlation with house prices.
- FullBath also contributes significantly to predicting house prices.
- BedroomAbvGr has a comparatively weaker relationship with house prices.
- The model explains approximately 63% of the variance in house prices using the selected features.

## Project Structure

```text
PRODIGY_DS_01
│
├── train.csv
├── house_price_prediction.py
├── README.md
├── correlation_heatmap.png
├── actual_vs_predicted.png
└── feature_importance.png
```

## How to Run

1. Install required libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

2. Place train.csv in the project directory.

3. Run the script

```bash
python house_price_prediction.py
```

4. View the generated visualizations and evaluation metrics.

## Conclusion

A Linear Regression model was successfully developed to predict house prices using selected housing features. The project demonstrates the complete machine learning workflow, including data preprocessing, model training, evaluation, and visualization.
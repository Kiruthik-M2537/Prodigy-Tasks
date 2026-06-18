# Customer Segmentation using K-Means Clustering

## Overview

This project applies the K-Means Clustering algorithm to segment customers based on their annual income and spending behavior. Customer segmentation helps businesses understand different customer groups and create targeted marketing strategies.

## Objective

To group retail store customers into meaningful segments based on:

- Annual Income (k$)
- Spending Score (1–100)

## Dataset

Source:
https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

Dataset File:
- Mall_Customers.csv

## Technologies Used

- Python
- Pandas
- Matplotlib
- Scikit-Learn

## Methodology

### Data Preparation

Selected the following features:

- Annual Income (k$)
- Spending Score (1–100)

### Elbow Method

Used the Elbow Method to determine the optimal number of clusters.

Optimal Clusters Found:
- K = 5

### K-Means Clustering

Applied K-Means clustering with 5 clusters and visualized customer groups along with cluster centroids.

## Results

Successfully segmented customers into five distinct groups:

1. High Income - High Spending
2. High Income - Low Spending
3. Low Income - High Spending
4. Low Income - Low Spending
5. Average Income - Average Spending

## Business Insights

- Premium customers can be targeted with exclusive offers.
- High-income low-spending customers have potential for increased engagement.
- Loyal high-spending customers should be retained through rewards programs.
- Budget-conscious customers can be targeted with discounts and promotions.

## Visualizations

### Elbow Method
Determines the optimal number of clusters.

### Customer Segmentation
Displays customer groups and cluster centroids.

## Project Structure

```text
PRODIGY_DS_02
│
├── Mall_Customers.csv
├── customer_segmentation.py
├── README.md
│
└── outputs
    ├── elbow_method.png
    └── customer_segments.png
```

## How to Run

```bash
pip install pandas matplotlib scikit-learn
python customer_segmentation.py
```

## Conclusion

K-Means Clustering successfully identified five meaningful customer segments. These insights can help businesses improve customer targeting, marketing campaigns, and overall customer relationship management.
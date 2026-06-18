import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load Dataset
df = pd.read_csv("Mall_Customers.csv")

# Select Features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Create Output Folder
import os
os.makedirs("outputs", exist_ok=True)

# -----------------------------
# Elbow Method
# -----------------------------
wcss = []

for i in range(1, 11):
    kmeans = KMeans(
        n_clusters=i,
        init='k-means++',
        random_state=42,
        n_init=10
    )
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.savefig('outputs/elbow_method.png')
plt.show()

# -----------------------------
# K-Means Clustering
# -----------------------------
kmeans = KMeans(
    n_clusters=5,
    init='k-means++',
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X)

# Add Cluster Labels
df['Cluster'] = clusters

# -----------------------------
# Customer Segmentation Plot
# -----------------------------
plt.figure(figsize=(10, 7))

plt.scatter(
    X.iloc[:, 0],
    X.iloc[:, 1],
    c=clusters,
    cmap='viridis',
    s=60
)

plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=300,
    marker='X',
    label='Centroids'
)

plt.title('Customer Segmentation using K-Means')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()

plt.savefig('outputs/customer_segments.png')
plt.show()

# -----------------------------
# Cluster Summary
# -----------------------------
print("\nCluster Summary:")
print(
    df.groupby('Cluster')[
        ['Annual Income (k$)', 'Spending Score (1-100)']
    ].mean()
)

print("\nTask Completed Successfully!")
print("Plots saved in outputs folder.")
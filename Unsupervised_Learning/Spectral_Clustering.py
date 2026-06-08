import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler

# Step 1: Load CSV data
df = pd.read_csv('spectralcluster.csv')

# Step 2: Extract features
X = df.values  # Assumes all columns are features

# Step 3: Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Apply Spectral Clustering
n_clusters = 2  # Set number of clusters you want
spectral_model = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', assign_labels='kmeans')
labels = spectral_model.fit_predict(X_scaled)

# Step 5: Add labels to DataFrame
df['Cluster'] = labels

# Step 6: Plot the clusters (for 2D data)
plt.figure(figsize=(8, 6))
for cluster in range(n_clusters):
    cluster_points = df[df['Cluster'] == cluster]
    plt.scatter(cluster_points.iloc[:, 0], cluster_points.iloc[:, 1], label=f'Cluster {cluster}')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Spectral Clustering Result')
plt.legend()
plt.grid(True)
plt.show()

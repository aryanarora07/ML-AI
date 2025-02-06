import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA

# Step 1: Load the dataset
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
df = pd.read_csv('iris.data.txt', header=None, names=column_names)

# Step 2: Preprocess the data (Drop class column & Normalize features)
features = df.drop(columns=['class'])
scaler = StandardScaler()
data_normalized = scaler.fit_transform(features)

# Step 3: Apply K-Means Clustering
# Find optimal K using the Elbow Method
inertia = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(data_normalized)
    inertia.append(kmeans.inertia_)

# Plot Elbow Method Graph
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, marker='o', linestyle='--')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.show()

# Fit K-Means with optimal K (3 for Iris dataset)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['KMeans_Cluster'] = kmeans.fit_predict(data_normalized)

# Step 4: Apply Hierarchical Clustering
linked = linkage(data_normalized, method='ward')
plt.figure(figsize=(10, 5))
dendrogram(linked, labels=df.index, leaf_rotation=90, leaf_font_size=8)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Data Points')
plt.ylabel('Euclidean Distance')
plt.show()

# Apply Agglomerative Clustering
hc = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
df['Hierarchical_Cluster'] = hc.fit_predict(data_normalized)

# Step 5: Evaluate Clustering Quality
silhouette_kmeans = silhouette_score(data_normalized, df['KMeans_Cluster'])
silhouette_hc = silhouette_score(data_normalized, df['Hierarchical_Cluster'])

print(f"Silhouette Score for K-Means: {silhouette_kmeans:.3f}")
print(f"Silhouette Score for Hierarchical Clustering: {silhouette_hc:.3f}")

# Adjusted Rand Index (ARI) - compares with true labels
true_labels = pd.factorize(df['class'])[0]
ari_kmeans = adjusted_rand_score(true_labels, df['KMeans_Cluster'])
ari_hc = adjusted_rand_score(true_labels, df['Hierarchical_Cluster'])

print(f"Adjusted Rand Index for K-Means: {ari_kmeans:.3f}")
print(f"Adjusted Rand Index for Hierarchical Clustering: {ari_hc:.3f}")

# Step 6: Visualizing Clusters using PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(data_normalized)
df['PC1'] = pca_data[:, 0]
df['PC2'] = pca_data[:, 1]

plt.figure(figsize=(8, 5))
sns.scatterplot(x=df['PC1'], y=df['PC2'], hue=df['KMeans_Cluster'], palette='viridis')
plt.title('K-Means Clustering (PCA Reduced)')
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(x=df['PC1'], y=df['PC2'], hue=df['Hierarchical_Cluster'], palette='viridis')
plt.title('Hierarchical Clustering (PCA Reduced)')
plt.show()

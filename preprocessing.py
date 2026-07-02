# preprocessing.py

import pandas as pd
import numpy as np
import pickle
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------
# STEP 1: Load Raw Data
# ---------------------------------------------------
print("Loading raw data...")
df = pd.read_csv("data/online_retail.csv", encoding="ISO-8859-1")
print(f"Raw data shape: {df.shape}")

# ---------------------------------------------------
# STEP 2: Data Cleaning
# ---------------------------------------------------
print("\nCleaning data...")

# 2a. Remove rows with missing CustomerID
# Without a CustomerID we can't attribute a purchase to anyone,
# so these rows are useless for segmentation and recommendation.
df = df.dropna(subset=["CustomerID"])

# 2b. Remove cancelled invoices
# Cancelled invoices have InvoiceNo starting with 'C' (e.g. 'C536379').
# We convert InvoiceNo to string first so we can check its first character.
df["InvoiceNo"] = df["InvoiceNo"].astype(str)
df = df[~df["InvoiceNo"].str.startswith("C")]

# 2c. Remove negative or zero Quantity
# Negative quantity = a return. Zero quantity = no real transaction.
df = df[df["Quantity"] > 0]

# 2d. Remove negative or zero UnitPrice
# A price of 0 or less isn't a real sale (could be a data entry error).
df = df[df["UnitPrice"] > 0]

# 2e. Drop rows with missing Description (product name)
# We need product names for the recommendation system to work.
df = df.dropna(subset=["Description"])

# 2f. Convert InvoiceDate to proper datetime format
# Right now it's just text. We need real datetime objects to calculate Recency.
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# 2g. Create a TotalPrice column (needed for Monetary calculation)
df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

print(f"Cleaned data shape: {df.shape}")
print(f"Rows removed: {541909 - df.shape[0]}")

# Save the cleaned dataset so we don't need to repeat this step
df.to_csv("data/clean_data.csv", index=False)
print("Saved clean_data.csv")

# ---------------------------------------------------
# STEP 3: RFM Feature Engineering
# ---------------------------------------------------
print("\nCalculating RFM values...")

# Recency needs a reference date — we use one day after the
# last transaction in the entire dataset as "today".
reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
print(f"Reference date for Recency: {reference_date}")

# Group everything by CustomerID and calculate R, F, M in one go
rfm = df.groupby("CustomerID").agg(
    Recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
    Frequency=("InvoiceNo", "nunique"),
    Monetary=("TotalPrice", "sum")
).reset_index()

print(f"RFM table shape: {rfm.shape}")
print(rfm.describe())

# ---------------------------------------------------
# STEP 4: Scale RFM Values
# ---------------------------------------------------
print("\nScaling RFM values...")

# KMeans is distance-based, so features must be on the same scale.
# Without scaling, Monetary (which can be in thousands) would dominate
# Recency (which is just in days), making clustering meaningless.
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

# ---------------------------------------------------
# STEP 5: Find Optimal Number of Clusters
# ---------------------------------------------------
print("\nFinding optimal K using Elbow Method + Silhouette Score...")

inertia_values = []
silhouette_values = []
k_range = range(2, 11)

for k in k_range:
    kmeans_test = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans_test.fit_predict(rfm_scaled)
    inertia_values.append(kmeans_test.inertia_)
    silhouette_values.append(silhouette_score(rfm_scaled, labels))
    print(f"K={k} | Inertia={kmeans_test.inertia_:.2f} | Silhouette={silhouette_values[-1]:.4f}")

# Save these for the Elbow chart in Streamlit
elbow_data = pd.DataFrame({
    "K": list(k_range),
    "Inertia": inertia_values,
    "Silhouette": silhouette_values
})
elbow_data.to_csv("data/elbow_data.csv", index=False)
print("Saved elbow_data.csv")

# ---------------------------------------------------
# STEP 6: Train Final KMeans Model
# ---------------------------------------------------
# Based on the project's segment table (High-Value, Regular, Occasional,
# At-Risk), we use K=4 clusters to match these 4 business labels.
FINAL_K = 4
print(f"\nTraining final KMeans model with K={FINAL_K}...")

kmeans_model = KMeans(n_clusters=FINAL_K, random_state=42, n_init=10)
rfm["Cluster"] = kmeans_model.fit_predict(rfm_scaled)

final_silhouette = silhouette_score(rfm_scaled, rfm["Cluster"])
print(f"Final Silhouette Score: {final_silhouette:.4f}")

# ---------------------------------------------------
# STEP 7: Label Clusters with Business Meaning
# ---------------------------------------------------
print("\nLabeling clusters...")

# We look at the average RFM values per cluster, then rank them
# to decide which cluster is High-Value vs At-Risk etc.
# Logic: High-Value = low Recency (bought recently), high Frequency, high Monetary.
cluster_summary = rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()
print(cluster_summary)

# Create a combined score: low Recency is good, high Frequency/Monetary is good.
# We rank each cluster on all three and combine the ranks.
cluster_summary["Recency_Rank"] = cluster_summary["Recency"].rank(ascending=True)   # lower recency = better = rank 1
cluster_summary["Frequency_Rank"] = cluster_summary["Frequency"].rank(ascending=False)  # higher freq = better = rank 1
cluster_summary["Monetary_Rank"] = cluster_summary["Monetary"].rank(ascending=False)    # higher monetary = better = rank 1
cluster_summary["Overall_Rank"] = (
    cluster_summary["Recency_Rank"] + cluster_summary["Frequency_Rank"] + cluster_summary["Monetary_Rank"]
)

# Sort clusters from best (High-Value) to worst (At-Risk)
sorted_clusters = cluster_summary.sort_values("Overall_Rank").index.tolist()

segment_labels = ["High-Value", "Regular", "Occasional", "At-Risk"]
cluster_to_segment = {cluster_id: segment_labels[i] for i, cluster_id in enumerate(sorted_clusters)}

print("Cluster to Segment mapping:", cluster_to_segment)

rfm["Segment"] = rfm["Cluster"].map(cluster_to_segment)

print("\nSegment distribution:")
print(rfm["Segment"].value_counts())

# ---------------------------------------------------
# STEP 8: Build Product Recommendation System
# ---------------------------------------------------
print("\nBuilding product similarity matrix...")

# Create a CustomerID x Description pivot table.
# Each cell = total quantity that customer bought of that product.
# This is the "purchase history matrix" mentioned in the project.
customer_product_matrix = df.pivot_table(
    index="CustomerID",
    columns="Description",
    values="Quantity",
    aggfunc="sum",
    fill_value=0
)

print(f"Customer-Product matrix shape: {customer_product_matrix.shape}")

# Item-based collaborative filtering: we compare PRODUCTS to each other
# (not customers), based on which customers bought them.
# So we transpose the matrix first (products as rows).
product_customer_matrix = customer_product_matrix.T

# Cosine similarity tells us how similar two products are based on
# the pattern of which customers bought them and how much.
similarity_matrix = cosine_similarity(product_customer_matrix)

# Instead of storing the full NxN matrix (120MB+), we store only the
# top 20 most similar products per product as a compact dictionary.
# This reduces the file from 114MB to ~3MB — well under GitHub's 100MB limit.
TOP_N = 20
similarity_compact = {}
products_list = product_customer_matrix.index.tolist()

print(f"Building compact top-{TOP_N} similarity lookup...")
for i, product in enumerate(products_list):
    scores_row = pd.Series(similarity_matrix[i], index=products_list)
    scores_row = scores_row.drop(index=product)  # exclude self-similarity
    top = scores_row.nlargest(TOP_N)
    similarity_compact[product] = list(zip(top.index.tolist(), [float(v) for v in top.values]))

similarity_df = similarity_compact  # keep variable name consistent
print(f"Compact dict: {len(similarity_compact)} products x top {TOP_N} each")

# ---------------------------------------------------
# STEP 9: Save Everything for Streamlit
# ---------------------------------------------------
print("\nSaving all models and data...")

pickle.dump(kmeans_model, open("models/kmeans_model.pkl", "wb"))
pickle.dump(scaler, open("models/scaler.pkl", "wb"))
pickle.dump(cluster_to_segment, open("models/cluster_to_segment.pkl", "wb"))
pickle.dump(rfm, open("models/rfm_data.pkl", "wb"))
pickle.dump(similarity_compact, open("models/similarity_compact.pkl", "wb"))

print("\nAll done! Files saved:")
print("  data/clean_data.csv")
print("  data/elbow_data.csv")
print("  models/kmeans_model.pkl")
print("  models/scaler.pkl")
print("  models/cluster_to_segment.pkl")
print("  models/rfm_data.pkl")
print("  models/similarity_compact.pkl")

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px

# Judul Web
st.title("Analisis Klaster Kemiskinan di Provinsi Sumatera")
st.markdown("Menggunakan PCA dan K-Means Clustering berdasarkan data persentase penduduk miskin.")

# Load Data
data = pd.read_csv('persentase-penduduk-miskin-menurut-provinsi-di-pulau-sumatera (1).csv', delimiter=';')
data_pivot = data.pivot_table(
    index='bps_nama_provinsi',
    columns=['tahun', 'semester'],
    values='persentase_penduduk_miskin_sumatera'
)
data_pivot = data_pivot.interpolate(axis=1).fillna(method='bfill', axis=1).fillna(method='ffill', axis=1)

# Normalisasi
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_pivot)

# K-Means dan PCA
optimal_clusters = 4
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
clusters = kmeans.fit_predict(data_scaled)
data_pivot['Cluster'] = clusters

pca = PCA(n_components=2)
pca_data = pca.fit_transform(data_scaled)
df_pca = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
df_pca['Cluster'] = clusters
df_pca['Provinsi'] = data_pivot.index.values

centroid_pca = pca.transform(kmeans.cluster_centers_)

# --- Streamlit Visualisasi ---
# Visualisasi Heatmap
st.subheader("Heatmap Tren Kemiskinan Provinsi Sumatera")
fig1, ax1 = plt.subplots(figsize=(10, 6))
sns.heatmap(data_pivot.drop('Cluster', axis=1), cmap='coolwarm', ax=ax1)
st.pyplot(fig1)

# Visualisasi PCA Interaktif
st.subheader("Visualisasi 2D Klaster Hasil PCA")
fig2 = px.scatter(
    df_pca, x='PC1', y='PC2', color=df_pca['Cluster'].astype(str),
    hover_data=['Provinsi'],
    title="PCA Clustering Visualization"
)
st.plotly_chart(fig2)

# Tampilkan isi masing-masing cluster
st.subheader("Detail Provinsi per Cluster")
for i in range(optimal_clusters):
    prov = df_pca[df_pca['Cluster'] == i]['Provinsi'].tolist()
    st.markdown(f"**Cluster {i}** ({len(prov)} provinsi):")
    st.write(", ".join(prov))
    st.markdown("---")

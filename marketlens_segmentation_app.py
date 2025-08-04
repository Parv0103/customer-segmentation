import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import plotly.express as px

# Streamlit Configuration
st.set_page_config(page_title="MarketLens", layout="wide")
st.title("MarketLens: Analytical Segmentation Framework")
st.markdown("""
### Customer Segmentation Dashboard
This tool clusters mall customers into distinct groups based on spending behavior and demographics.
""")

# Load and display data
@st.cache_data
def load_data():
    return pd.read_csv('Mall_Customers.csv')

customer_data = load_data()

# Data Exploration Section
st.header("🔍 Data Exploration")
with st.expander("View Raw Data"):
    st.dataframe(customer_data)

# Feature Selection
features = st.multiselect(
    "Select features for clustering:",
    options=['Age', 'Annual Income (k$)', 'Spending Score (1-100)'],
    default=['Annual Income (k$)', 'Spending Score (1-100)']
)

# Visualization Type Toggle
st.header("🎚️ Visualization Options")
viz_type = st.radio(
    "Select visualization type:",
    ("2D", "3D"),
    horizontal=True,
    index=0
)

# Data Preprocessing
X = customer_data[features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Optimal Cluster Detection
st.header("📊 Optimal Cluster Detection")
max_clusters = st.slider("Maximum clusters to test:", 2, 10, 5)

wcss = []
silhouette_scores = []
cluster_range = range(2, max_clusters + 1)
for i in cluster_range:
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Elbow Plot
fig1, ax1 = plt.subplots()
ax1.plot(cluster_range, wcss, marker='o', color='orange')
ax1.set_title('Figure 4.1: Elbow Method Plot for Optimal Clusters')
ax1.set_xlabel('Number of Clusters')
ax1.set_ylabel('Within-Cluster Sum of Squares (WCSS)')
st.pyplot(fig1)

# Optional Silhouette Score Plot
show_silhouette = st.checkbox("Show Silhouette Score vs Cluster Count")
if show_silhouette:
    fig_silhouette, ax_sil = plt.subplots()
    ax_sil.plot(cluster_range, silhouette_scores, marker='s', linestyle='--', color='green')
    ax_sil.set_title('Silhouette Score vs Cluster Count')
    ax_sil.set_xlabel('Number of Clusters')
    ax_sil.set_ylabel('Silhouette Score')
    st.pyplot(fig_silhouette)

# Cluster Execution
optimal_clusters = st.number_input("Select number of clusters:", 2, 10, 5)
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(X_scaled)
customer_data['Cluster'] = clusters
centroids_original_scale = scaler.inverse_transform(kmeans.cluster_centers_)

# Visualization Section
st.header("📈 Cluster Visualization")

if viz_type == "2D":
    if len(features) >= 2:
        fig2, ax2 = plt.subplots(figsize=(8,6))
        scatter = ax2.scatter(X[:,0], X[:,1], c=clusters, cmap='viridis', s=50)
        ax2.scatter(centroids_original_scale[:,0], centroids_original_scale[:,1], 
                    s=200, c='red', marker='X', label='Centroids')
        ax2.set_title(f'2D Cluster Visualization ({features[0]} vs {features[1]})')
        ax2.set_xlabel(features[0])
        ax2.set_ylabel(features[1])
        ax2.legend()
        st.pyplot(fig2)
    else:
        st.warning("Select at least 2 features for 2D visualization")

elif viz_type == "3D":
    if len(features) == 3:
        fig3 = px.scatter_3d(
            customer_data,
            x=features[0],
            y=features[1],
            z=features[2],
            color='Cluster',
            title="Figure 4.3: 3D Cluster Visualization",
            height=600
        )
        centroids_df = pd.DataFrame(centroids_original_scale, columns=features)
        centroids_df['Cluster'] = range(optimal_clusters)
        fig3.add_scatter3d(
            x=centroids_df[features[0]],
            y=centroids_df[features[1]],
            z=centroids_df[features[2]],
            mode='markers',
            marker=dict(size=10, color='red', symbol='x'),
            name='Centroids'
        )
        st.plotly_chart(fig3)
    else:
        st.warning("Select exactly 3 features for 3D visualization")

# PCA Visualization
st.header("📉 PCA Visualization (2D)")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
fig_pca, ax_pca = plt.subplots()
scatter = ax_pca.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='cool')
ax_pca.set_title("Figure 4.2: PCA Visualization of Customer Segments")
ax_pca.set_xlabel("PCA Component 1")
ax_pca.set_ylabel("PCA Component 2")
st.pyplot(fig_pca)

# Cluster Insights Section
st.header("💡 Cluster Insights")
cluster_profiles = []
for cluster in range(optimal_clusters):
    cluster_data = customer_data[customer_data['Cluster'] == cluster]
    profile = {
        "Cluster": cluster,
        "Size": len(cluster_data),
        "Avg Age": cluster_data['Age'].mean(),
        "Avg Income": cluster_data['Annual Income (k$)'].mean(),
        "Avg Spending": cluster_data['Spending Score (1-100)'].mean()
    }
    if profile["Avg Income"] > 70 and profile["Avg Spending"] > 60:
        profile["Label"] = "Premium Spenders"
        profile["Recommendation"] = "Target with luxury products and VIP services"
    elif profile["Avg Income"] > 70 and profile["Avg Spending"] < 40:
        profile["Label"] = "Wealthy Conservatives"
        profile["Recommendation"] = "Focus on trust-building and value propositions"
    elif profile["Avg Income"] < 40 and profile["Avg Spending"] > 60:
        profile["Label"] = "Budget Enthusiasts"
        profile["Recommendation"] = "Offer discounts and loyalty programs"
    else:
        profile["Label"] = "Average Customers"
        profile["Recommendation"] = "General marketing approaches"
    cluster_profiles.append(profile)

# Display profiles
for profile in cluster_profiles:
    with st.expander(f"Cluster {profile['Cluster']}: {profile['Label']} ({profile['Size']} customers)"):
        cols = st.columns(3)
        cols[0].metric("Avg Age", f"{profile['Avg Age']:.1f} years")
        cols[1].metric("Avg Income", f"${profile['Avg Income']:.1f}k")
        cols[2].metric("Avg Spending", f"{profile['Avg Spending']:.1f}/100")
        st.markdown(f"**Business Recommendation:** {profile['Recommendation']}")
        sample_customers = customer_data[customer_data['Cluster'] == profile['Cluster']].sample(3)
        st.write("Sample customers:", sample_customers[['CustomerID', 'Gender', 'Age']])

# Key Findings
st.header("🔑 Key Findings")
st.markdown("""
1. **Customer Segments Identified**: The analysis reveals distinct customer groups based on spending patterns
2. **Actionable Insights**: Each cluster has clear characteristics that enable targeted marketing
3. **Resource Allocation**: Helps prioritize marketing efforts toward high-value segments
4. **Personalization Opportunities**: Enables tailored product recommendations for each segment
""")

# How to Use These Insights
st.header("🚀 How to Use These Insights")
st.markdown("""
- **Premium Spenders**: Focus on high-margin products and exclusive offers
- **Wealthy Conservatives**: Emphasize quality, durability, and investment value
- **Budget Enthusiasts**: Promote deals, bundles, and value propositions
- **Average Customers**: Use broad marketing with opportunities for upselling
""")

# Export Option
if st.button("Export Cluster Data"):
    customer_data.to_csv('customer_segments.csv', index=False)
    st.success("Cluster data exported successfully!")

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import davies_bouldin_score, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF

# Load datasets
customers_df = pd.read_csv('../Customers.csv')
transactions_df = pd.read_csv('../Transactions.csv')

# Merge customer data with transaction data
merged_data = transactions_df.merge(customers_df, on='CustomerID', how='left')

# Aggregate transaction data to create customer profiles (sum of quantities and total value)
customer_profiles = (
    merged_data.groupby('CustomerID')
    .agg({
        'Quantity': 'sum',
        'TotalValue': 'sum',
    })
    .reset_index()
)

# Normalize numerical features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(customer_profiles[['Quantity', 'TotalValue']])

# Apply KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42)  # You can choose between 2 and 10 clusters
customer_profiles['Cluster'] = kmeans.fit_predict(scaled_data)

# Calculate DB Index
db_index = davies_bouldin_score(scaled_data, customer_profiles['Cluster'])

# Calculate Silhouette Score
silhouette_avg = silhouette_score(scaled_data, customer_profiles['Cluster'])

# Calculate Inertia (sum of squared distances of samples to their cluster center)
inertia = kmeans.inertia_

# Visualize the clusters using original scaled features (Quantity and TotalValue)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=customer_profiles['Quantity'], y=customer_profiles['TotalValue'], hue=customer_profiles['Cluster'], palette='viridis', s=100)
plt.title('Customer Segmentation using KMeans Clustering')
plt.xlabel('Total Quantity Purchased')
plt.ylabel('Total Value of Transactions')
plt.legend(title='Cluster')

# Save the plot as an image file for embedding into the PDF
plt.savefig('customer_clusters.png', bbox_inches='tight')
plt.close()

# Generate PDF report
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()

# Title
pdf.set_font("Arial", size=16, style='B')
pdf.cell(200, 10, txt="Customer Segmentation Report", ln=True, align='C')

# Clustering Info
pdf.ln(10)  # Line break
pdf.set_font("Arial", size=12)
pdf.multi_cell(0, 10, txt=f"Number of Clusters: 4\n")
pdf.multi_cell(0, 10, txt=f"DB Index: {db_index:.4f}\n")
pdf.multi_cell(0, 10, txt=f"Silhouette Score: {silhouette_avg:.4f}\n")
pdf.multi_cell(0, 10, txt=f"Inertia (Within-cluster sum of squares): {inertia:.4f}\n")

# Embed the clustering plot
pdf.ln(10)  # Line break
pdf.cell(200, 10, txt="Cluster Visualization:", ln=True)
pdf.ln(5)  # Line break
pdf.image('customer_clusters.png', x=10, y=pdf.get_y(), w=180)

# Save the PDF file
pdf.output("customer_segmentation_report.pdf")

print("PDF report generated successfully!")

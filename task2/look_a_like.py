import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load datasets
customers_df = pd.read_csv('../Customers.csv')
products_df = pd.read_csv('../Products.csv')
transactions_df = pd.read_csv('../Transactions.csv')

# Convert dates to datetime format
customers_df['SignupDate'] = pd.to_datetime(customers_df['SignupDate'])
transactions_df['TransactionDate'] = pd.to_datetime(transactions_df['TransactionDate'])

# Merge transaction and product data to get detailed transaction info
merged_data = transactions_df.merge(products_df, on='ProductID', how='left')

# Aggregate transactions to create customer profiles
customer_profiles = (
    merged_data.groupby('CustomerID')
    .agg({
        'Quantity': 'sum',  # Total quantity purchased
        'TotalValue': 'sum',  # Total transaction value
        'Category': lambda x: ','.join(x),  # Concatenate purchased categories
    })
    .reset_index()
)

# Merge with customer data
customer_profiles = customer_profiles.merge(customers_df, on='CustomerID', how='left')

# Handle numerical and categorical features
numerical_features = ['Quantity', 'TotalValue']
categorical_features = ['Region', 'Category']

# Apply preprocessing: scale numerical features, one-hot encode categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ]
)

# Transform the data
X = preprocessor.fit_transform(customer_profiles)

# Calculate cosine similarity
similarity_matrix = cosine_similarity(X)

# Get top 3 similar customers for each of the first 20 customers
def get_top_lookalikes(customer_id, similarity_matrix, customer_profiles, top_n=3):
    customer_index = customer_profiles.index[customer_profiles['CustomerID'] == customer_id].tolist()[0]
    customer_similarities = similarity_matrix[customer_index]
    
    # Exclude self-similarity
    customer_similarities[customer_index] = -1
    
    # Get top N most similar customers
    top_indices = np.argsort(customer_similarities)[-top_n:][::-1]
    top_customers = customer_profiles.iloc[top_indices]['CustomerID']
    top_scores = customer_similarities[top_indices]
    
    return list(zip(top_customers, top_scores))

# Filter customers to include only the first 20 (CustomerID: C0001 to C0020)
top_20_customers = customers_df[customers_df['CustomerID'].isin([f'C{i:04d}' for i in range(1, 21)])]

# Generate lookalike recommendations for customers C0001 - C0020
lookalikes = {}
for customer_id in top_20_customers['CustomerID']:
    lookalikes[customer_id] = get_top_lookalikes(customer_id, similarity_matrix, customer_profiles)

# Create Lookalike.csv in the required format
lookalike_map = []
for cust_id, similar_customers in lookalikes.items():
    similar_customers_list = [f"({sim_cust_id}, {round(score, 4)})" for sim_cust_id, score in similar_customers]
    similar_customers_str = f"[{', '.join(similar_customers_list)}]"
    lookalike_map.append({
        'CustomerID': cust_id,
        'SimilarCustomers': similar_customers_str,
    })

lookalike_df = pd.DataFrame(lookalike_map)
lookalike_df.to_csv('Lookalike.csv', index=False)

print("Lookalike.csv has been generated successfully!")

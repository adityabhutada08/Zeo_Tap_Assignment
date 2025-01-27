import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF

# Load the datasets
customers_df = pd.read_csv('Customers.csv')
products_df = pd.read_csv('Products.csv')
transactions_df = pd.read_csv('Transactions.csv')

# Convert date columns to datetime format for proper analysis
customers_df['SignupDate'] = pd.to_datetime(customers_df['SignupDate'])
transactions_df['TransactionDate'] = pd.to_datetime(transactions_df['TransactionDate'])

# ===== DATA CLEANING =====
# Drop rows with missing values to ensure data consistency
customers_df.dropna(inplace=True)
products_df.dropna(inplace=True)
transactions_df.dropna(inplace=True)

# Remove duplicate rows to avoid redundant records
customers_df.drop_duplicates(inplace=True)
products_df.drop_duplicates(inplace=True)
transactions_df.drop_duplicates(inplace=True)

# ===== EXPLORATORY DATA ANALYSIS =====
# Set the visualization style for consistent plots
sns.set(style="whitegrid")

# 1. Analyze the distribution of customers by region
region_distribution = customers_df['Region'].value_counts()
most_common_region = region_distribution.idxmax()  # Find the region with the most customers
plt.figure(figsize=(8, 5))
sns.countplot(x='Region', data=customers_df, palette="viridis")
plt.title("Customer Distribution by Region", fontsize=14)
plt.xlabel("Region", fontsize=12)
plt.ylabel("Number of Customers", fontsize=12)
plt.xticks(rotation=45)
plt.savefig("customer_distribution.png")  # Save the plot
plt.close()

# 2. Analyze signup trends over time
customers_df['SignupYearMonth'] = customers_df['SignupDate'].dt.to_period('M')  # Extract year-month for trend analysis
signup_trends = customers_df['SignupYearMonth'].value_counts().sort_index()
plt.figure(figsize=(12, 6))
signup_trends.plot(kind="line", marker='o', color="teal")
plt.title("Customer Signup Trends Over Time", fontsize=14)
plt.xlabel("Year-Month", fontsize=12)
plt.ylabel("Number of Signups", fontsize=12)
plt.grid(True)
plt.savefig("signup_trends.png")  # Save the plot
plt.close()

# 3. Analyze top-performing product categories by total sales
# Merge transactions and products data to include category information
merged_data = transactions_df.merge(products_df, on='ProductID', how='left')
category_sales = merged_data.groupby('Category')['TotalValue'].sum().sort_values(ascending=False)
top_category = category_sales.idxmax()  # Identify the category with the highest sales
plt.figure(figsize=(10, 6))
sns.barplot(x=category_sales.index, y=category_sales.values, palette="magma")
plt.title("Top-Performing Product Categories by Sales", fontsize=14)
plt.xlabel("Category", fontsize=12)
plt.ylabel("Total Sales (USD)", fontsize=12)
plt.xticks(rotation=45)
plt.savefig("top_categories.png")  # Save the plot
plt.close()

# 4. Identify the most purchased products by total quantity sold
product_purchases = merged_data.groupby('ProductName')['Quantity'].sum().sort_values(ascending=False)
top_product = product_purchases.idxmax()  # Find the most purchased product
plt.figure(figsize=(10, 6))
sns.barplot(x=product_purchases.values[:10], y=product_purchases.index[:10], palette="coolwarm")
plt.title("Top 10 Most Purchased Products", fontsize=14)
plt.xlabel("Total Quantity Sold", fontsize=12)
plt.ylabel("Product Name", fontsize=12)
plt.savefig("top_products.png")  # Save the plot
plt.close()

# 5. Analyze monthly transaction trends
transactions_monthly = transactions_df['TransactionDate'].dt.to_period('M').value_counts().sort_index()
peak_month = transactions_monthly.idxmax()  # Identify the month with the highest transaction count
plt.figure(figsize=(12, 6))
transactions_monthly.plot(kind="line", marker='o', color="purple")
plt.title("Monthly Transaction Trends", fontsize=14)
plt.xlabel("Month", fontsize=12)
plt.ylabel("Number of Transactions", fontsize=12)
plt.grid(True)
plt.savefig("monthly_transactions.png")  # Save the plot
plt.close()

# ===== PDF REPORT CREATION =====
# Define a PDF class with custom headers and footers
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'EDA Report - eCommerce Transactions', border=0, ln=1, align='C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 10)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

pdf = PDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()

# Add an introduction to the PDF report
pdf.set_font('Arial', '', 12)
pdf.multi_cell(0, 10, "This report contains an exploratory data analysis of eCommerce transaction data, providing insights into customer behavior, product performance, and sales trends.")

# Add a section for insights
pdf.ln(10)
pdf.set_font('Arial', 'B', 12)
pdf.cell(0, 10, "Key Insights:", ln=1)

# Generate insights dynamically based on the EDA results
detailed_insights = []

# Customer distribution insights
region_counts = customers_df['Region'].value_counts()
most_common_region = region_counts.idxmax()
least_common_region = region_counts.idxmin()
detailed_insights.append(
    f"1. Customer Distribution by Region: The '{most_common_region}' region has the highest number of customers ({region_counts[most_common_region]}), "
    f"while the '{least_common_region}' region has the lowest ({region_counts[least_common_region]})."
)

# Signup trends insights
peak_signup_month = signup_trends.idxmax()
detailed_insights.append(
    f"2. Signup Trends: The peak signup month was {peak_signup_month}, with {signup_trends[peak_signup_month]} customers signing up."
)

# Product category performance insights
detailed_insights.append(
    f"3. Top Product Category: The '{top_category}' category generated the highest revenue (${category_sales[top_category]:,.2f})."
)

# Popular products insights
top_product_name = product_purchases.idxmax()
detailed_insights.append(
    f"4. Most Popular Product: '{top_product_name}' was the most purchased product, with {product_purchases[top_product_name]} units sold."
)

# Monthly transactions insights
peak_month = transactions_monthly.idxmax()
detailed_insights.append(
    f"5. Monthly Transactions: The highest transaction volume occurred in {peak_month}."
)

# Write insights into the PDF
pdf.set_font('Arial', '', 12)
for insight in detailed_insights:
    pdf.ln()
    pdf.multi_cell(0, 10, insight)

# Add plots to the PDF
pdf.add_page()
pdf.set_font('Arial', 'B', 12)
pdf.cell(0, 10, "Visualizations:", ln=1)

# Add each plot image to the PDF
plots = [
    ("customer_distribution.png", "Customer Distribution by Region"),
    ("signup_trends.png", "Customer Signup Trends Over Time"),
    ("top_categories.png", "Top-Performing Product Categories by Sales"),
    ("top_products.png", "Top 10 Most Purchased Products"),
    ("monthly_transactions.png", "Monthly Transaction Trends")
]

for plot_path, title in plots:
    pdf.ln(10)
    pdf.cell(0, 10, title, ln=1)
    pdf.image(plot_path, x=10, y=None, w=180)

# Save the PDF report
pdf.output("EDA_Report.pdf")
print("\nPDF report 'EDA_Report.pdf' has been generated.")

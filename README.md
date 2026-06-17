
# Customer Segmentation and Churn Risk Analysis using K-Means Clustering

## Project Overview

Businesses often treat all customers the same, even though customer behavior varies significantly. Some customers purchase frequently and generate substantial revenue, while others make a single purchase and never return.

In this project, I analyzed an online retail transaction dataset and built a customer segmentation framework using behavioral features such as purchase recency, spending patterns, customer lifespan, and purchase frequency.

The objective was to identify meaningful customer groups, understand their contribution to revenue, and evaluate their potential churn risk. The final segmentation was performed using K-Means Clustering and visualized using Principal Component Analysis (PCA).

---

## Business Problem

The company has thousands of customers but limited information about:

- Which customers generate the most revenue?
- Which customers are likely to stop purchasing?
- Which customer groups should receive retention campaigns?
- Which customers should be prioritized for loyalty programs?

Customer segmentation helps answer these questions by grouping customers with similar purchasing behavior.

---

## Dataset

**Dataset:** Online Retail Dataset

The dataset contains transaction-level records including:

- Customer ID
- Invoice Number
- Invoice Date
- Quantity
- Unit Price
- Country

Each row represents a product purchased by a customer.

---

## Feature Engineering

Customer-level features were created from raw transaction data:

| Feature | Description |
|----------|-------------|
| Recency | Days since last purchase |
| Monetary | Total amount spent |
| Total Invoice | Total purchase frequency |
| Customer Span | Active duration of customer |
| Gap Days | Average gap between purchases |
| Avg Expense Per Invoice | Average spending per order |

These features were used to represent customer behavior before clustering.

---

## Methodology

### 1. Data Cleaning

- Removed missing Customer IDs
- Handled transaction anomalies
- Investigated customers with negative monetary values caused by returns/refunds

### 2. Feature Scaling

Since clustering is distance-based, all features were standardized using StandardScaler.

### 3. Customer Segmentation

K-Means Clustering was applied to identify customer groups.

Cluster count selection was performed using:

- Elbow Method
- Silhouette Score

**Final Number of Clusters:** 8

### 4. Segment Profiling

Each cluster was analyzed and assigned a business-friendly name based on purchasing behavior.

### 5. PCA Visualization

Principal Component Analysis (PCA) was applied to reduce the feature space into two dimensions for visualization.

The first two principal components retained approximately **61.05% of the total variance**, allowing meaningful visualization of customer segments.

---

## Customer Segments Identified

### VIP Customers

- High spending customers
- Low churn risk
- Ideal candidates for loyalty programs

### High-Value Repeat Customers

- Largest revenue contributors
- Frequent purchasers
- Core customer base

### Enterprise Customers

- Extremely high-value accounts
- Distinct purchasing behavior

### At-Risk Customers

- Previously active customers
- Increasing inactivity
- High churn probability

### Lost One-Time Buyers

- Low engagement
- High churn risk
- Potential reactivation targets

---

## Key Findings

### Revenue Distribution

- High-Value Repeat Customers generated the largest share of total revenue.
- VIP and Enterprise Customers contributed significant business value despite smaller customer counts.

### Churn Analysis

- At-Risk Customers exhibited the highest churn rates.
- Lost One-Time Buyers represented customers requiring re-engagement campaigns.

### Outlier Detection

A single customer with negative monetary value was isolated into an independent cluster.

Further investigation revealed that this was caused by product returns/refunds rather than normal purchasing behavior.

### PCA Insights

PCA visualization showed clear separation between several customer segments, validating the effectiveness of the clustering process.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
- Scikit-Learn
- K-Means Clustering
- Principal Component Analysis (PCA)

---

## Project Structure

```text
Customer-Segmentation/
│
├── notebooks/
│   ├── 01_Customer_Segmentation.ipynb
│   └── 02_Experiments.ipynb
│
├── figures/
│   ├── elbow_method.png
│   ├── silhouette_score.png
│   └── pca_visualization.png
│
├── README.md
│
└── requirements.txt
```

---

## Future Improvements

- Compare K-Means with DBSCAN and Hierarchical Clustering
- Build segment-specific churn prediction models
- Deploy an interactive dashboard using Streamlit
- Perform real-time customer segmentation

---

## Conclusion

This project demonstrates how customer transaction data can be transformed into actionable business insights using machine learning.

By combining feature engineering, K-Means clustering, churn analysis, and PCA visualization, meaningful customer segments were identified that can support targeted marketing, customer retention strategies, and revenue optimization.

The analysis highlights how unsupervised learning can be used to uncover hidden customer patterns and drive data-informed business decisions.

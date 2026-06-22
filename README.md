
<div align="center">

<!-- HEADER BANNER -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=Customer%20Segmentation%20%26%20Churn%20Analysis&fontSize=32&fontColor=ffffff&animation=twinkling&fontAlignY=38&desc=Online%20Retail%20II%20%C2%B7%20RFM%20Features%20%C2%B7%20K-Means%20Clustering&descAlignY=58&descSize=16" width="100%"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-Viz-4C72B0?style=for-the-badge)](https://seaborn.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-Complete-22C55E?style=for-the-badge)]()

<br/>

> *"Transforming 1 million raw transactions into 8 actionable customer stories."*

<br/>

**Made with 💙 by [Chandraveer Singh](https://github.com/chandraveersingh)**  
🎓 **Indian Institute of Technology (BHU), Varanasi**

</div>

---

## 👋 About This Project

This is a complete, end-to-end Data Analysis project built on a real-world e-commerce dataset containing over **1 million transactions** from a UK-based online retailer.

I built this because the standard "just run KMeans on RFM" approach felt too shallow for a real business problem. A customer who buys every 60 days and hasn't returned in 80 days is very different from one who buys quarterly and is only at 80 days — yet a simple recency threshold treats them identically. That bothered me, so I designed a **personalised, behaviour-driven churn framework** that accounts for each customer's unique buying rhythm.

The result: **26.3% estimated churn rate**, **8 interpretable customer segments**, and a methodology that actually means something to a real business team.

---

## 📋 Table of Contents

- [Project Highlights](#-project-highlights)
- [Dataset](#-dataset)
- [Project Architecture](#-project-architecture)
- [Feature Engineering](#-feature-engineering)
- [Churn Identification Logic](#-churn-identification-logic)
- [Customer Segments](#-customer-segments)
- [Key Results](#-key-results)
- [Visualisations](#-visualisations)
- [How to Run](#-how-to-run)
- [Tech Stack](#-tech-stack)
- [Author](#-author)

---

## ✨ Project Highlights

<div align="center">

| 🎯 | What Makes This Different |
|----|--------------------------|
| 🧠 | **Behaviour-driven churn** — not a static 90-day cutoff, but a dynamic per-customer threshold |
| 📐 | **Novel `gap_days` feature** — captures a customer's natural buying rhythm, not just frequency |
| 🔬 | **Principled missing value handling** — NaN in `gap_days` is *informative*, not just noise |
| 📊 | **Dual cluster validation** — Elbow Method + Silhouette Score for statistically sound K selection |
| 💼 | **Business-first framing** — every technical decision tied back to a real business implication |

</div>

---

## 📦 Dataset

| Attribute | Details |
|-----------|---------|
| **Name** | Online Retail II |
| **Source** | [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Online+Retail+II) |
| **Period** | 01 Dec 2009 – 09 Dec 2011 |
| **Size** | ~1,067,371 transactions |
| **Scope** | UK-based non-store online retailer (gift ware) |

**Schema**

| Column | Type | Description |
|--------|------|-------------|
| `Invoice` | str | Invoice number; prefix `C` = cancellation/return |
| `StockCode` | str | Product code |
| `Description` | str | Product name |
| `Quantity` | int | Units purchased (negative = return) |
| `InvoiceDate` | datetime | Date and time of transaction |
| `Price` | float | Unit price (GBP £) |
| `Customer ID` | float | Unique customer identifier |
| `Country` | str | Customer's country |

---

## 🏗️ Project Architecture

```
online_retail_II.xlsx
        │
        ▼
┌─────────────────────┐
│   Data Cleaning     │  Remove duplicates, nulls, adjustment entries
│   & Preprocessing   │  Engineer Total = Price × Quantity
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Feature            │  Recency, Frequency, Monetary, Avg Order Value,
│  Engineering        │  Customer Span, gap_days (purchase rhythm)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Churn              │  Dynamic per-customer threshold (blends individual
│  Identification     │  behaviour with population median)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  EDA & Correlation  │  Distributions, outlier analysis, heatmap
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  K-Means            │  StandardScaler → Elbow + Silhouette → K=8
│  Clustering         │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Segment Profiles   │  8 labelled customer segments with business
│  & PCA Viz          │  recommendations and PCA biplot
└─────────────────────┘
```

---

## 🛠️ Feature Engineering

Starting from raw transactions, I build a **customer-level feature table** (one row per customer):

<div align="center">

```
┌──────────────────────────────────────────────────────────────────┐
│              CUSTOMER-LEVEL FEATURES                             │
├────────────────────────────┬─────────────────────────────────────┤
│ Recency                    │ Days since last purchase            │
│ Total Invoice (Frequency)  │ Number of distinct orders placed    │
│ Monetary                   │ Total £ spend across all orders     │
│ Avg Expense per Invoice    │ Average order value (£)             │
│ Customer Span              │ Days between first & last purchase  │
│ gap_days ⭐                │ Average days between purchases      │
│ single_purchase_customer   │ Binary: only 1 order ever?          │
└────────────────────────────┴─────────────────────────────────────┘
```

</div>

### ⭐ The `gap_days` Feature

This is the most analytically interesting feature in the project. Instead of just counting orders, I computed each customer's **average time between consecutive purchases**:

```python
# 1. Deduplicate to invoice-level timeline
af = df.groupby(["Invoice", "Customer ID"])["InvoiceDate"].min().reset_index()

# 2. Sort chronologically per customer
af = af.sort_values(["Customer ID", "InvoiceDate"])

# 3. Compute gap between consecutive purchases
af["Next_Date"] = af.groupby("Customer ID")["InvoiceDate"].shift(-1)
af["gap_days"]  = (af["Next_Date"] - af["InvoiceDate"]).dt.total_seconds() / 86400

# 4. Average gap per customer
avg_gap = af.groupby("Customer ID")["gap_days"].mean()
```

> **Why this matters:** A customer with `gap_days = 14` who hasn't purchased in 50 days is in a very different situation than one with `gap_days = 60` who hasn't purchased in 50 days. Frequency alone can't capture this.

---

## 🧠 Churn Identification Logic

This is the heart of the project. Instead of a naive "inactive for X days = churned" rule, I designed a **two-path churn framework**:

### Path 1 — Repeat Customers

$$\text{Dynamic Threshold} = 0.85 \times \overline{\text{gap}}_{\text{customer}} + 0.15 \times \widetilde{\text{gap}}_{\text{global}}$$

$$\text{Churn if: Recency} > 2 \times \text{Dynamic Threshold}$$

- The **0.85/0.15 blend** prevents extreme individual gaps from dominating while still personalising the model.
- The **2× multiplier** adds a grace period, reducing false positives for customers who are slightly late relative to their cycle.

### Path 2 — Single-Purchase Customers

$$\text{Churn if: Recency} > 2.5 \times P_{85}(\text{gap\_days})$$

- Uses the **85th percentile** of all purchase gaps as a conservative population benchmark.
- Avoids flagging recently acquired one-time buyers.

### Final Flag

```python
Potential_Churn = (single_visit_churn == 1) OR (repeat_churn == 1)
```

<div align="center">

```
4,365 Total Customers
├── 3,216 Active  (73.7%) ✅
└── 1,149 Potential Churn (26.3%) ⚠️
         ├── Lost One-Time Buyers     (never returned, long dormant)
         └── At-Risk Repeat Customers (exceeded personalised gap threshold)
```

</div>

---

## 🗂️ Customer Segments

After K-Means clustering (K=8, validated by Elbow + Silhouette), customers were grouped into 8 distinct segments:

<div align="center">

| Segment | Customers | Avg Recency | Avg Orders | Avg £ Spend | Churn Risk |
|---------|-----------|-------------|------------|-------------|------------|
| 💎 VIP Customers | Small | Very Low | Very High | Very High | 🟢 Minimal |
| 🏢 Enterprise Customers | Small | Low | Highest | Highest | 🟢 Very Low |
| 🔁 High-Value Repeat | Medium | Low | High | High | 🟢 Low |
| 🎯 Regular Active | Large | Low-Med | Moderate | Moderate | 🟢 Very Low |
| 🆕 Recent One-Time Buyers | Medium | Low | 1 | Low | 🟡 Watch |
| ⚠️ At-Risk Customers | Medium | High | Moderate | Moderate | 🔴 High |
| 👻 Lost One-Time Buyers | Medium | Very High | 1 | Low | 🔴 Very High |
| 🔍 Data Outlier | Tiny | — | — | Negative | ⚪ N/A |

</div>

### Business Actions by Segment

```
💎 VIP + 🏢 Enterprise  →  Dedicated account management · Exclusive access · SLA perks
🔁 High-Value Repeat    →  Loyalty programme · Early product access · Referral rewards
🎯 Regular Active       →  Cross-sell campaigns · Seasonal promotions · Upsell nudges
🆕 Recent One-Time      →  Welcome series · "Here's what you might like" email
⚠️ At-Risk              →  WIN-BACK NOW · Personalised discount · Re-engagement flow
👻 Lost One-Time        →  Final win-back attempt · Survey for exit feedback
```

---

## 📊 Key Results

<div align="center">

```
┌────────────────────────────────────────────────────────────────┐
│                    PROJECT RESULTS                             │
├─────────────────────────────┬──────────────────────────────────┤
│ Customers Analysed          │ 4,365                            │
│ Estimated Churn Rate        │ 26.3%                            │
│ Single-Purchase Customers   │ 1,253 (28.7%)                    │
│ Optimal Clusters            │ K = 8                            │
│ Silhouette Score @ K=8      │ > 0.40                           │
│ PCA Variance Retained       │ ~61%                             │
│ Strongest Churn Signal      │ Recency (corr: +0.61 with churn) │
│ Strongest Revenue Driver    │ Frequency (corr: +0.63 with £)   │
└─────────────────────────────┴──────────────────────────────────┘
```

### Correlation Highlights

| Feature Pair | Correlation | Insight |
|-------------|-------------|---------|
| Frequency ↔ Monetary | **+0.63** | More orders = more revenue; prioritise repeat purchase conversion |
| Recency ↔ Churn | **+0.61** | Inactivity is the top churn predictor |
| Customer Span ↔ Recency | **−0.50** | Long-tenure customers stay more active |
| Customer Span ↔ Frequency | **+0.46** | Older customers naturally buy more over time |
| Gap Days ↔ Churn | **−0.36** | Personalised thresholds protect slow-cycle buyers from false flags |

</div>

---

## 🔭 Next Steps

- [ ] Integrate churn scores into a CRM for automated re-engagement triggers
- [ ] A/B test win-back campaign effectiveness: At-Risk vs Lost One-Time Buyers
- [ ] Layer in product category data for deeper behavioural profiling
- [ ] Retrain segmentation model quarterly as new transactions arrive
- [ ] Build a Customer Lifetime Value (CLV) prediction model on these features
- [ ] Explore DBSCAN or GMM as alternative clustering approaches

---

## ▶️ How to Run

### Prerequisites

```bash
pip install pandas numpy scikit-learn seaborn matplotlib openpyxl
```

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/chandraveersingh/customer-segmentation-churn-analysis.git
cd customer-segmentation-churn-analysis

# 2. Place the dataset in the project root
# Download from: https://archive.ics.uci.edu/ml/datasets/Online+Retail+II
# File name: online_retail_II.xlsx

# 3. Launch Jupyter
jupyter notebook Customer_Segmentation_Churn_Analysis.ipynb
```

### File Structure

```
customer-segmentation-churn-analysis/
│
├── Customer_Segmentation_Churn_Analysis.ipynb   ← Main notebook
├── online_retail_II.xlsx                        ← Dataset (download separately)
├── requirements.txt                             ← Python dependencies
└── README.md                                    ← You are here
```

### `requirements.txt`

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
seaborn>=0.12.0
matplotlib>=3.6.0
openpyxl>=3.0.0
```

---

## 🧰 Tech Stack

<div align="center">

| Tool | Purpose |
|------|---------|
| ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) | Core language |
| ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) | Data manipulation & feature engineering |
| ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) | Numerical computing |
| ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) | KMeans, PCA, StandardScaler, Silhouette |
| ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat) | Base plotting |
| ![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=flat) | Statistical visualisations |
| ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white) | Interactive notebook environment |

</div>

---

## 👤 Author

<div align="center">

<br/>

### Chandraveer Singh

🎓 **Indian Institute of Technology (BHU), Varanasi**

*Data Analyst · Turning raw data into decisions that matter*

<br/>

[![GitHub](https://img.shields.io/badge/GitHub-chandraveersingh-181717?style=for-the-badge&logo=github)](https://github.com/chandraveersingh)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/chandraveersingh)

<br/>

*If you found this project useful or interesting, a ⭐ on the repo would mean a lot!*

</div>

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer" width="100%"/>

*Built with curiosity, coffee, and a stubborn refusal to use a 90-day churn cutoff.*

</div>

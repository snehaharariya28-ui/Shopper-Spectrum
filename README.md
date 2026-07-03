# 🛒 Shopper Spectrum
### Customer Segmentation & Product Recommendation System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://shopper-spectrum-2fjgeniqhcqeklusjabkfc.streamlit.app)

> An end-to-end e-commerce analytics dashboard that segments customers using RFM analysis and KMeans clustering, and recommends similar products using item-based collaborative filtering — built with Python and Streamlit.



## 🔗 Live Demo

**Streamlit App:** https://shopper-spectrum-2fjgeniqhcqeklusjabkfc.streamlit.app



##  About the Project

Shopper Spectrum is an end-to-end e-commerce analytics dashboard built on real-world retail transaction data from a UK-based online retail store, containing over 5 lakh transactions from customers across 37 countries.

The e-commerce industry generates massive amounts of transaction data every day. Most businesses collect this data but fail to extract meaningful insights from it. Shopper Spectrum bridges that gap — transforming raw transaction records into actionable customer intelligence and personalized product recommendations.

The project addresses two core business problems:

**1. Who are our customers?**
Customers are segmented based on their purchase behavior using RFM (Recency, Frequency, Monetary) analysis combined with KMeans clustering — classifying every customer into one of four actionable groups: High-Value, Regular, Occasional, and At-Risk. Each segment comes with a clear business interpretation and recommended action.

**2. What should we recommend?**
An item-based collaborative filtering recommendation system suggests 5 similar products for any given product — powered by cosine similarity computed on a customer-product purchase matrix. The system learns from actual purchase patterns, not product descriptions or categories.

Together, these two systems enable businesses to run targeted marketing campaigns, improve customer retention, and drive personalized shopping experiences at scale.



## Dataset

**Name:** Online Retail Dataset

**Original Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/352/online+retail)

**About the dataset:**
The original dataset contains transactional data from a UK-based online retail store. It covers all transactions between December 2010 and December 2011. The dataset was originally donated to the UCI Machine Learning Repository and is widely used for e-commerce analytics and customer segmentation research.

**Note:** This project was completed as part of a Data Analytics Internship. The dataset was provided with transaction dates modified to the period December 2022 to December 2023 to simulate a more recent business scenario. The structure, columns, and content of the data remain the same as the original UCI dataset.

**Dataset Details:**

| Column | Description |
|---|---|
| InvoiceNo | Unique transaction number |
| StockCode | Unique product code |
| Description | Name of the product |
| Quantity | Number of units purchased |
| InvoiceDate | Date and time of transaction |
| UnitPrice | Price per unit (in GBP £) |
| CustomerID | Unique identifier for each customer |
| Country | Country where the customer is based |

**Raw Data:** 5,41,909 rows across 8 columns

**After Cleaning:** 3,97,884 valid transactions



## Data Cleaning

The following steps were applied to clean the raw data:

- Removed rows with missing **CustomerID** (1,35,080 rows)
- Removed **cancelled invoices** (InvoiceNo starting with 'C')
- Removed rows with **negative or zero Quantity**
- Removed rows with **negative or zero UnitPrice**
- Removed rows with missing **product Description**
- Engineered a new column: **TotalPrice = Quantity × UnitPrice**



## Methodology

### 1. RFM Feature Engineering

For each customer, three values were calculated:

- **Recency (R)** — Number of days since their last purchase. Lower is better.
- **Frequency (F)** — Total number of unique transactions. Higher is better.
- **Monetary (M)** — Total amount spent. Higher is better.

### 2. Customer Segmentation (KMeans Clustering)

- RFM values were normalized using **StandardScaler**
- Optimal number of clusters was selected using **Elbow Method** and **Silhouette Score**
- **K=4** was chosen based on both evaluation metrics and business requirements
- Clusters were labeled based on their average RFM profiles:

| Segment | Recency | Frequency | Monetary | Description |
|---|---|---|---|---|
| High-Value | Very Low | Very High | Very High | Frequent, recent, big spenders |
| Regular | Low | Medium | Medium | Steady buyers, not premium |
| Occasional | Medium | Low | Low | Rare purchases |
| At-Risk | Very High | Very Low | Low | Haven't purchased in a long time |

### 3. Product Recommendation (Collaborative Filtering)

- A **CustomerID × Product** pivot matrix was built (4,338 customers × 3,877 products)
- **Cosine Similarity** was computed between all product pairs
- For each product, the **top 20 most similar products** are stored
- Two products are considered similar if the same customers frequently buy both



## Key Insights

- Total Revenue: **£8,911,408** across 18,532 transactions
- **United Kingdom** accounts for ~82% of total revenue (£7.3 million)
- Best selling product by revenue: **Paper Craft Little Birdie**
- Revenue peaks in **November 2023** due to holiday shopping season
- **High-Value segment** has only 13 customers but contributes **18.6% of revenue**
- **Occasional segment** is the largest group (3,054 customers, 46.6% of revenue)
- **1,067 At-Risk customers** represent over £5 lakh in recoverable revenue



## Streamlit Application — Pages

| Page | Description |
|---|---|
| Home | KPI overview and segment summary |
| Sales Analytics | Revenue trends, top products, transaction distribution |
| Country Analysis | Revenue and transactions by country |
| RFM Analysis | Distribution of R, F, M values and segment-wise box plots |
| Cluster Selection | Elbow Method and Silhouette Score charts |
| Customer Segmentation | 2D and 3D cluster visualizations |
| Similarity Matrix | Product similarity bar chart for any selected product |
| Product Recommendation | Input product → get 5 similar products |
| Predict Segment | Input RFM values → get predicted customer segment |
| Business Insights | Key findings and revenue contribution by segment |



## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data cleaning and feature engineering |
| Scikit-learn | KMeans clustering, StandardScaler, Cosine Similarity |
| Plotly | Interactive charts and visualizations |
| Streamlit | Web application framework |
| Pickle | Saving and loading trained models |



## ⚙️ Setup Instructions

**1. Install dependencies:**
```
pip install -r requirements.txt
```

**2. Run preprocessing** (cleans data, trains models, saves pkl files):
```
python preprocessing.py
```

**3. Launch the app:**
```
streamlit run app.py
```



##  Project Structure

```
shopper_spectrum/
│
├── data/
│   ├── online_retail.csv          # raw dataset
│   ├── clean_data.csv             # generated by preprocessing.py
│   └── elbow_data.csv             # generated by preprocessing.py
│
├── models/
│   ├── kmeans_model.pkl           # trained KMeans model
│   ├── scaler.pkl                 # StandardScaler
│   ├── cluster_to_segment.pkl     # cluster to segment name mapping
│   ├── rfm_data.pkl               # RFM dataframe with segment labels
│   └── similarity_compact.pkl     # top 20 similar products per product
│
├── app.py                         # Streamlit application (10 pages)
├── preprocessing.py               # data cleaning + model training
├── requirements.txt               # Python dependencies
├── .gitignore
└── README.md
```

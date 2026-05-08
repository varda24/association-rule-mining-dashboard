# Smart Market Basket Analysis and Product Recommendation System

## Live Demo

🔗 https://aappciation-rule-mining-dashboard-6pzcwy6zygpodcbz5fcqjm.streamlit.app/

## Project Overview

The Smart Market Basket Analysis System is a Machine Learning-based web application that analyzes grocery transaction datasets to discover relationships between products using the Apriori Algorithm and Association Rule Mining.

The system helps identify:

- Frequently purchased product combinations
- Cross-selling opportunities
- Product placement suggestions
- Customer buying patterns

Users can upload their own grocery transaction datasets and instantly receive intelligent recommendations through an interactive dashboard.

---

# Features

- Upload custom grocery datasets
- Association Rule Mining using Apriori Algorithm
- Product recommendation generation
- Cross-selling insights
- Product placement suggestions
- Dark-themed modern dashboard UI
- Strong, Medium, and Weak association classification
- Duplicate recommendation filtering
- Responsive and user-friendly interface

---

# Technologies Used

| Technology | Purpose |
|------------|----------|
| Python | Backend Development |
| Flask | Web Framework |
| Pandas | Data Processing |
| Mlxtend | Apriori Algorithm |
| HTML | Frontend Structure |
| CSS | Styling and UI |
| Association Rule Mining | Machine Learning Concept |

---

# Machine Learning Concepts Used

## Association Rule Mining

Association Rule Mining is an unsupervised machine learning technique used to identify relationships between products in transaction datasets.

### Example

Customers buying bread often buy butter.

---

## Apriori Algorithm

The Apriori Algorithm identifies:

- Frequent itemsets
- Product associations
- Customer purchasing patterns

### Working Process

1. Finding frequent products
2. Generating association rules
3. Measuring rule strength using support, confidence, and lift

---

# Evaluation Metrics

## Support

Measures how frequently products appear together.

Formula:

Support(A → B) = Transactions containing A and B / Total Transactions

---

## Confidence

Measures the probability that customers buying A also buy B.

Formula:

Confidence(A → B) = Support(A ∪ B) / Support(A)

---

## Lift

Measures the strength of association between products.

Formula:

Lift(A → B) = Confidence(A → B) / Support(B)

---

# Project Structure

```bash
association-rule-project/
│
├── app.py
├── model.py
├── requirements.txt
│
├── dataset/
│      └── groceries.csv
│
├── templates/
│      └── index.html
│
└── static/
       └── style.css
```

---

# Installation Steps

## 1. Clone Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_LINK
```

---

## 2. Open Project Folder

```bash
cd association-rule-project
```

---

## 3. Create Virtual Environment

```bash
python -m venv venv
```

---

## 4. Activate Virtual Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

---

## 5. Install Required Libraries

```bash
pip install -r requirements.txt
```

---

# Running the Project

Run the Flask server:

```bash
python app.py
```

Open browser:

```text
http://127.0.0.1:5000
```

---

# Dataset Format

The dataset should contain transaction records like:

```text
milk,bread,butter
bread,eggs
whole milk,yogurt
```

Each line represents one customer transaction.

---

# Example Output

## Product Placement

- Place citrus fruit near tropical fruit shelves.

## Cross Selling

- Customers buying butter often purchase whipped/sour cream.

## Customer Buying Pattern

- 24.2% of customers buying fruit/vegetable juice also purchase yogurt.

---

# Future Enhancements

- Data visualization charts
- Recommendation analytics dashboard
- Download recommendation reports
- User authentication system
- Real-time recommendation engine
- FP-Growth algorithm implementation
- Cloud database integration

---

# Applications

- Supermarket analytics
- E-commerce recommendation systems
- Retail business intelligence
- Customer purchasing analysis
- Product placement optimization

---

# Author

**Varda Kunde**  
CSE(AIML) Student  
DIEMS


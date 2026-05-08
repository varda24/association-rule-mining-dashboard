import streamlit as st
import pandas as pd

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(
    page_title="Smart Market Basket Analysis",
    layout="wide"
)

# -----------------------------
# Custom CSS
# -----------------------------

st.markdown("""
<style>

.main {
    background-color: #0f172a;
    color: white;
}

h1 {
    color: #38bdf8;
    text-align: center;
}

.stFileUploader {
    background-color: #1e293b;
    padding: 20px;
    border-radius: 10px;
}

.card {
    background-color: #1e293b;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 15px;
    border-left: 6px solid #38bdf8;
}

.strong {
    color: #22c55e;
    font-weight: bold;
}

.medium {
    color: #f59e0b;
    font-weight: bold;
}

.weak {
    color: #94a3b8;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Title
# -----------------------------

st.title("🛒 Smart Market Basket Analysis")

st.markdown("""
Upload your grocery dataset and discover intelligent
product pairings using Association Rule Mining and
Apriori Algorithm.
""")

# -----------------------------
# File Upload
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload Grocery Dataset",
    type=["csv", "txt"]
)

# -----------------------------
# Processing
# -----------------------------

if uploaded_file:

    transactions = []

    content = uploaded_file.read().decode("utf-8")

    lines = content.splitlines()

    for line in lines:

        items = line.strip().split(",")

        transactions.append(items)

    # Transaction Encoding
    te = TransactionEncoder()

    te_array = te.fit(transactions).transform(transactions)

    df = pd.DataFrame(te_array, columns=te.columns_)

    # Apriori
    frequent_items = apriori(
        df,
        min_support=0.005,
        use_colnames=True
    )

    rules = association_rules(
        frequent_items,
        metric="confidence",
        min_threshold=0.2
    )

    if rules.empty:

        st.warning(
            "No strong product relationships found."
        )

    else:

        rules = rules.sort_values(
            by="lift",
            ascending=False
        )

        st.subheader("📊 Product Recommendations")

        used_pairs = set()

        count = 0

        for _, row in rules.iterrows():

            antecedent = list(row['antecedents'])[0]
            consequent = list(row['consequents'])[0]

            if antecedent == consequent:
                continue

            pair = tuple(
                sorted([antecedent, consequent])
            )

            if pair in used_pairs:
                continue

            used_pairs.add(pair)

            confidence = round(
                row['confidence'] * 100,
                1
            )

            # Strong
            if confidence >= 40:

                category = "🛒 Product Placement"

                text = (
                    f"Place {antecedent} near "
                    f"{consequent} shelves."
                )

                badge = "🔥 Strong Association"

                badge_class = "strong"

            # Medium
            elif confidence >= 25:

                category = "📈 Cross Selling"

                text = (
                    f"Customers buying "
                    f"{antecedent} often "
                    f"purchase {consequent}."
                )

                badge = "⚡ Medium Association"

                badge_class = "medium"

            # Weak
            else:

                category = "📊 Customer Buying Pattern"

                text = (
                    f"{confidence}% of customers "
                    f"buying {antecedent} also "
                    f"purchase {consequent}."
                )

                badge = "🔹 Weak Association"

                badge_class = "weak"

            st.markdown(f"""
            <div class="card">

            <h4>{category}</h4>

            <p>{text}</p>

            <p class="{badge_class}">
            {badge}
            </p>

            </div>
            """, unsafe_allow_html=True)

            count += 1

            if count >= 10:
                break
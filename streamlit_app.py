import streamlit as st
import pandas as pd

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Smart Market Basket Analysis",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

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

.card h4 {
    color: #38bdf8;
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

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.title("🛒 Smart Market Basket Analysis")

st.markdown("""
Upload your grocery dataset and discover intelligent
product pairings using Association Rule Mining and
Apriori Algorithm.
""")

# ---------------------------------------------------
# FILE UPLOADER
# ---------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Grocery Dataset",
    type=["csv", "txt"]
)

# ---------------------------------------------------
# PROCESSING
# ---------------------------------------------------

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

    df = pd.DataFrame(
        te_array,
        columns=te.columns_
    )

    # Apriori Algorithm
    frequent_items = apriori(
        df,
        min_support=0.005,
        use_colnames=True
    )

    # Generate Rules
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

        # Sort by lift
        rules = rules.sort_values(
            by="lift",
            ascending=False
        )

        st.subheader("📊 Product Recommendations")

        used_pairs = set()

        recommendations = []

        for _, row in rules.iterrows():

            antecedent = list(row['antecedents'])[0]
            consequent = list(row['consequents'])[0]

            # Skip same products
            if antecedent == consequent:
                continue

            # Remove duplicate pairs
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

            # ---------------------------------------------------
            # STRONG ASSOCIATION
            # ---------------------------------------------------

            if confidence >= 40:

                category = "🛒 Product Placement"

                text = (
                    f"Place {antecedent} near "
                    f"{consequent} shelves."
                )

                badge = "🔥 Strong Association"

                badge_class = "strong"

                priority = 1

            # ---------------------------------------------------
            # MEDIUM ASSOCIATION
            # ---------------------------------------------------

            elif confidence >= 25:

                category = "📈 Cross Selling"

                text = (
                    f"Customers buying "
                    f"{antecedent} often "
                    f"purchase {consequent}."
                )

                badge = "⚡ Medium Association"

                badge_class = "medium"

                priority = 2

            # ---------------------------------------------------
            # WEAK ASSOCIATION
            # ---------------------------------------------------

            else:

                category = "📊 Customer Buying Pattern"

                text = (
                    f"{confidence}% of customers "
                    f"buying {antecedent} also "
                    f"purchase {consequent}."
                )

                badge = "🔹 Weak Association"

                badge_class = "weak"

                priority = 3

            recommendations.append({

                "category": category,
                "text": text,
                "badge": badge,
                "badge_class": badge_class,
                "priority": priority

            })

            # Limit recommendations
            if len(recommendations) >= 10:
                break

        # ---------------------------------------------------
        # SORT RECOMMENDATIONS
        # ---------------------------------------------------

        recommendations = sorted(
            recommendations,
            key=lambda x: x["priority"]
        )

        # ---------------------------------------------------
        # DISPLAY RECOMMENDATIONS
        # ---------------------------------------------------

        for item in recommendations:

            st.markdown(f"""
            <div class="card">

            <h4>{item['category']}</h4>

            <p>{item['text']}</p>

            <p class="{item['badge_class']}">
            {item['badge']}
            </p>

            </div>
            """, unsafe_allow_html=True)
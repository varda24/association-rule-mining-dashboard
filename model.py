from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

def generate_rules(filepath):

    transactions = []

    with open(filepath, "r", encoding="utf-8") as file:

        for line in file:

            line = line.strip()

            if line == "":
                continue

            items = line.split(",")

            transactions.append(items)

    te = TransactionEncoder()

    te_array = te.fit(transactions).transform(transactions)

    df = pd.DataFrame(te_array, columns=te.columns_)

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

    return rules
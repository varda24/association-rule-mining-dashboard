from flask import Flask, render_template, request
import os
from model import generate_rules

app = Flask(__name__)

UPLOAD_FOLDER = "dataset"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/", methods=["GET", "POST"])
def home():

    recommendations = []

    used_pairs = set()

    if request.method == "POST":

        file = request.files["file"]

        if file:

            filepath = os.path.join(
                app.config["UPLOAD_FOLDER"],
                file.filename
            )

            file.save(filepath)

            # Generate association rules
            rules = generate_rules(filepath)

            # If no rules found
            if rules.empty:

                recommendations.append({
                    "text": "No strong product relationships found in the dataset.",
                    "strength": "🔹 Weak Association",
                    "priority": 3,
                    "category": "📊 Analysis Result"
                })

            else:

                # Sort by lift
                top_rules = rules.sort_values(
                    by="lift",
                    ascending=False
                )

                for _, row in top_rules.iterrows():

                    antecedent_list = list(row['antecedents'])
                    consequent_list = list(row['consequents'])

                    antecedent = antecedent_list[0]
                    consequent = consequent_list[0]

                    # Skip same products
                    if antecedent == consequent:
                        continue

                    # Remove duplicate product pairs
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

                    # -----------------------------------
                    # Strong Recommendations
                    # -----------------------------------

                    if confidence >= 40:

                        category = "🛒 Product Placement"

                        text = (
                            f"Place {antecedent} near "
                            f"{consequent} shelves."
                        )

                        strength = "🔥 Strong Association"

                        priority = 1

                    # -----------------------------------
                    # Medium Recommendations
                    # -----------------------------------

                    elif confidence >= 25:

                        category = "📈 Cross Selling"

                        text = (
                            f"Customers buying {antecedent} "
                            f"often purchase {consequent}."
                        )

                        strength = "⚡ Medium Association"

                        priority = 2

                    # -----------------------------------
                    # Weak Recommendations
                    # -----------------------------------

                    else:

                        category = "📊 Customer Buying Pattern"

                        text = (
                            f"{confidence}% of customers buying "
                            f"{antecedent} also purchase "
                            f"{consequent}."
                        )

                        strength = "🔹 Weak Association"

                        priority = 3

                    recommendation = {
                        "text": text,
                        "strength": strength,
                        "priority": priority,
                        "category": category
                    }

                    recommendations.append(recommendation)

                    # Limit recommendations
                    if len(recommendations) >= 10:
                        break

    # Sort recommendations
    recommendations = sorted(
        recommendations,
        key=lambda x: x["priority"]
    )

    return render_template(
        "index.html",
        recommendations=recommendations
    )


if __name__ == "__main__":
    app.run(debug=True)
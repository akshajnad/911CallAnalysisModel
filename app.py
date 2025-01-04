import os
import subprocess

import pandas as pd
from flask import Flask, request, render_template
from mlxtend.frequent_patterns import apriori, association_rules


app = Flask(__name__)
app.config["SECRET_KEY"] = "some_secret_key_for_session"


DATASET_ID = "mchirico/montcoalert"  # Kaggle dataset name
FILE_NAME = "911.csv"

if not os.path.exists(FILE_NAME):
    print(f"'{FILE_NAME}' not found. Downloading from Kaggle...")

  
    subprocess.run([
        "kaggle", "datasets", "download", "-d", DATASET_ID,
        "--unzip", "-p", "."
    ], check=True)

    if os.path.exists(FILE_NAME):
        print(f"Successfully downloaded '{FILE_NAME}'.")
    else:
        raise FileNotFoundError(
            f"Failed to download {FILE_NAME}. Please check Kaggle credentials."
        )

else:
    print(f"'{FILE_NAME}' is already present. Skipping download.")


###############################################################################
# 3) LOAD THE 911 DATA
###############################################################################
df_911 = pd.read_csv(FILE_NAME)


if "title" in df_911.columns and "Reason" not in df_911.columns:
    df_911["Reason"] = df_911["title"].apply(lambda x: x.split(":")[0] if pd.notnull(x) else x)
elif "Reason" in df_911.columns:
    df_911.rename(columns={"Reason": "reason"}, inplace=True)

if "timeStamp" in df_911.columns:
    df_911["timeStamp"] = pd.to_datetime(df_911["timeStamp"], errors='coerce')

if "reason" not in df_911.columns:
    df_911["reason"] = df_911["title"].apply(lambda x: x.split(":")[0] if pd.notnull(x) else x)

df_911 = df_911.dropna(subset=["reason"])

df_911["reason"] = df_911["reason"].astype("category")


df_onehot = pd.get_dummies(df_911["reason"], prefix="", prefix_sep="")

frequent_itemsets = apriori(df_onehot, min_support=0.01, use_colnames=True)

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
rules["antecedents_str"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
rules["consequents_str"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))

rules = rules.sort_values(by="confidence", ascending=False).reset_index(drop=True)


@app.route("/", methods=["GET", "POST"])
def index():
    # Unique reasons
    unique_reasons = sorted(df_911["reason"].unique())

    if request.method == "POST":
        selected_reason = request.form.get("selected_reason", "All")

        if selected_reason and selected_reason != "All":
            # Filter the 911 dataframe by that reason
            filtered_df = df_911[df_911["reason"] == selected_reason]
        else:
            filtered_df = df_911

        top_rules = rules.head(10)

        return render_template(
            "index.html",
            reasons=unique_reasons,
            selected_reason=selected_reason,
            table_data=filtered_df.head(100).to_html(
                classes="table table-striped",
                index=False
            ),
            rules_data=top_rules.to_html(
                classes="table table-bordered",
                index=False
            )
        )

    # Default GET: no filter
    selected_reason = "All"
    filtered_df = df_911
    top_rules = rules.head(10)

    return render_template(
        "index.html",
        reasons=unique_reasons,
        selected_reason=selected_reason,
        table_data=filtered_df.head(100).to_html(
            classes="table table-striped",
            index=False
        ),
        rules_data=top_rules.to_html(
            classes="table table-bordered",
            index=False
        )
    )


if __name__ == "__main__":
    # Start the 911-Association-Explorer in debug mode
    app.run(debug=True)

import os
import subprocess

import pandas as pd
from flask import Flask, request, render_template
from mlxtend.frequent_patterns import apriori, association_rules

###############################################################################
# 1) APPLICATION SETUP
###############################################################################
app = Flask(__name__)
app.config["SECRET_KEY"] = "some_secret_key_for_session"

###############################################################################
# 2) CHECK AND DOWNLOAD THE KAGGLE DATASET IF NEEDED
#    - We'll download "911.csv" from the Kaggle dataset "mchirico/montcoalert"
#    - This ensures the user does not have to manually download the file.
###############################################################################
DATASET_ID = "mchirico/montcoalert"  # Kaggle dataset name
FILE_NAME = "911.csv"

if not os.path.exists(FILE_NAME):
    print(f"'{FILE_NAME}' not found. Downloading from Kaggle...")

    # Execute Kaggle CLI to download the dataset and unzip it right here.
    # Make sure you have kaggle.json in ~/.kaggle or %USERPROFILE%\.kaggle
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

# OPTIONAL: Limit the data for demonstration if desired; otherwise keep it all
# df_911 = df_911.head(10000)

###############################################################################
# 4) BASIC PREPROCESSING
#    - Convert 'timeStamp' to datetime if available
#    - Extract 'reason' from the "title" column if needed
###############################################################################
# The dataset has a 'title' column like "EMS: BACK PAINS/INJURY"
# We'll parse out 'reason' if it's not already present.

if "title" in df_911.columns and "Reason" not in df_911.columns:
    df_911["Reason"] = df_911["title"].apply(lambda x: x.split(":")[0] if pd.notnull(x) else x)
elif "Reason" in df_911.columns:
    # If the Kaggle dataset already has a 'Reason' column, use that
    df_911.rename(columns={"Reason": "reason"}, inplace=True)

# Convert timeStamp if it exists
if "timeStamp" in df_911.columns:
    df_911["timeStamp"] = pd.to_datetime(df_911["timeStamp"], errors='coerce')

# If reason column doesn't exist, create it from the 'title' parse
if "reason" not in df_911.columns:
    df_911["reason"] = df_911["title"].apply(lambda x: x.split(":")[0] if pd.notnull(x) else x)

# Filter out any rows with NaN reason
df_911 = df_911.dropna(subset=["reason"])

# Convert reason to a category type
df_911["reason"] = df_911["reason"].astype("category")

###############################################################################
# 5) ASSOCIATION RULE MINING
#    - We'll do a simplistic approach, one-hot encoding the reason and
#      applying Apriori + association_rules.
###############################################################################

# One-hot encode "reason" -> multiple columns (EMS, Fire, Traffic, etc.)
df_onehot = pd.get_dummies(df_911["reason"], prefix="", prefix_sep="")

# Because the dataset can be large, let's set a somewhat moderate min_support
frequent_itemsets = apriori(df_onehot, min_support=0.01, use_colnames=True)

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
rules["antecedents_str"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
rules["consequents_str"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))

# Sort by confidence descending
rules = rules.sort_values(by="confidence", ascending=False).reset_index(drop=True)

###############################################################################
# 6) FLASK ROUTES
#    - Renders a page letting user filter by reason and see top association rules
###############################################################################
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

        # For demonstration, show top 10 rules
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

###############################################################################
# 7) RUN THE APPLICATION
###############################################################################
if __name__ == "__main__":
    # Start the 911-Association-Explorer in debug mode
    app.run(debug=True)

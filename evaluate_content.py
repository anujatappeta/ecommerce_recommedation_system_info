import pandas as pd
from cleaning_data import load_data, clean_data
from content_based import recommend_products

# Load and clean dataset
df_raw = load_data("clean_data.csv")
df = clean_data(df_raw)

df["Name"] = df["Name"].astype(str)
df["Brand"] = df["Brand"].astype(str)
df["Category"] = df["Category"].astype(str)


def evaluate_content(top_n=5):

    precision_scores = []
    recall_scores = []
    f1_scores = []

    for idx, row in df.iterrows():

        product_name = row["Name"]
        brand = row["Brand"]
        category = row["Category"]

        try:
            recommendations = recommend_products(product_name, top_n)
        except:
            continue

        if recommendations.empty:
            continue

        recommended_names = set(recommendations["Name"])

        # Ground truth (same brand OR same category)
        relevant_products = df[
            (
                (df["Brand"] == brand) |
                (df["Category"] == category)
            ) &
            (df["Name"] != product_name)
        ]["Name"]

        relevant_set = set(relevant_products)

        if len(relevant_set) == 0:
            continue

        true_positive = len(recommended_names & relevant_set)

        precision = true_positive / len(recommended_names)
        recall = true_positive / len(relevant_set)

        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    avg_precision = sum(precision_scores) / len(precision_scores)
    avg_recall = sum(recall_scores) / len(recall_scores)
    avg_f1 = sum(f1_scores) / len(f1_scores)

    print("Content-Based Recommendation Evaluation")
    print("--------------------------------------")
    print(f"Precision@{top_n}: {avg_precision:.4f}")
    print(f"Recall@{top_n}: {avg_recall:.4f}")
    print(f"F1-score@{top_n}: {avg_f1:.4f}")


if __name__ == "__main__":
    evaluate_content(top_n=5)

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from cleaning_data import load_data, clean_data
df_raw = load_data("clean_data.csv")
df = clean_data(df_raw)
df["Tags"] = df["Tags"].fillna("").astype(str)
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df["Tags"])
similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
def recommend_products(product_name, top_n=10):

    product_name = product_name.lower()

    matches = df[
        df["Name"].str.contains(product_name, na=False) |
        df["Tags"].str.contains(product_name, na=False)
    ]
    if matches.empty:
        print("Product not found!")
        return pd.DataFrame()
    index = matches.index[0]

    similarity_scores = list(enumerate(similarity_matrix[index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    similarity_scores = similarity_scores[1: top_n + 1]

    product_indices = [i[0] for i in similarity_scores]

    return df.iloc[product_indices][["Name", "Brand", "ReviewCount"]]
if __name__ == "__main__":

    result = recommend_products("LIPSTICK", top_n=5)

    print(result)

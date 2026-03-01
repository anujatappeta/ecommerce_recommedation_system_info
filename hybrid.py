import pandas as pd
from content_based import recommend_products as content_recommend
from collaborative_based import recommend_products as collaborative_recommend

def hybrid_recommend(user_id, product_name, top_n=10):

    content_results = content_recommend(product_name, top_n)
    collaborative_results = collaborative_recommend(user_id, top_n)

    combined = pd.concat([content_results, collaborative_results])
    combined = combined.drop_duplicates().head(top_n)

    return combined


if __name__ == "__main__":

    result = hybrid_recommend(user_id=1, product_name="LIPSTICK", top_n=5)

    print(result)
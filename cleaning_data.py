import pandas as pd

def load_data(file_path):
    return pd.read_csv(file_path)

def inspect_data(df):
    print("Initial DataFrame shape:", df.shape)
    print("\nDataFrame info:")
    print(df.info())
    print("\nDataFrame head:\n", df.head())

def clean_data(df, min_user_interactions=3, min_product_interactions=3):

    # Rename columns
    df.rename(columns={
        "User's ID": "UserID",
        "ProdID": "ProductID",
        "Review Count": "ReviewCount"
    }, inplace=True)

    # Convert IDs to numeric
    df["UserID"] = pd.to_numeric(df["UserID"], errors="coerce")
    df["ProductID"] = pd.to_numeric(df["ProductID"], errors="coerce")

    # Remove invalid IDs
    df = df[~df["UserID"].isin([0, -2147483648])]
    df = df[~df["ProductID"].isin([0, -2147483648])]

    df.dropna(subset=["UserID", "ProductID"], inplace=True)

    # Convert numeric columns
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    df["ReviewCount"] = pd.to_numeric(df["ReviewCount"], errors="coerce")

    df.dropna(subset=["Rating", "ReviewCount"], inplace=True)

    df["Rating"] = df["Rating"].astype(float)
    df["ReviewCount"] = df["ReviewCount"].astype(int)

    # Handle missing text fields
    df["Category"] = df["Category"].fillna(df["Tags"])
    df["Tags"] = df["Tags"].fillna(df["Category"])
    df["Brand"] = df["Brand"].fillna("Unknown")
    df.dropna(subset=["Name"], inplace=True)
    df["Description"] = df["Description"].fillna("")
    df["Tags"] = df["Tags"].fillna("")

    # Keep and clean ImageURL
    # Keep only the first image URL (remove everything after "|")
    if "ImageURL" in df.columns:
        df["ImageURL"] = df["ImageURL"].fillna("").astype(str)
        df["ImageURL"] = df["ImageURL"].str.split("|").str[0].str.strip()


    # Remove exact duplicate rows
    df.drop_duplicates(inplace=True)

    # Remove rating = 0
    df = df[df["Rating"] != 0]

    # Text normalization (fixed regex warnings)
    df["Category"] = df["Category"].str.lower().str.replace(r"[^\w\s,]", "", regex=True).str.strip()
    df["Tags"] = df["Tags"].str.lower().str.replace(r"[^\w\s,]", "", regex=True).str.strip()
    df["Brand"] = df["Brand"].str.lower().str.replace(r"[^\w\s]", "", regex=True).str.strip()
    df["Name"] = df["Name"].str.lower().str.replace(r"[^\w\s]", "", regex=True).str.strip()
    df["Description"] = df["Description"].str.lower().str.replace(r"[^\w\s]", "", regex=True).str.strip()

    # Filter users/products with too few interactions
    while True:
        user_counts = df["UserID"].value_counts()
        product_counts = df["ProductID"].value_counts()

        low_users = user_counts[user_counts < min_user_interactions].index
        low_products = product_counts[product_counts < min_product_interactions].index

        if len(low_users) == 0 and len(low_products) == 0:
            break

        df = df[~df["UserID"].isin(low_users)]
        df = df[~df["ProductID"].isin(low_products)]
    # Reset index
    df.reset_index(drop=True, inplace=True)

    return df

def save_clean_data(df, output_path):
    df.to_csv(output_path, index=False)
    
if __name__ == "__main__":
    data_path = "clean_data.csv"
    df_raw = load_data(data_path)
    df_cleaned = clean_data(df_raw)
    print("Cleaned DataFrame shape:", df_cleaned.shape)
    save_clean_data(df_cleaned, "cleaned_data.csv")
    print("Cleaning Completed Successfully.")

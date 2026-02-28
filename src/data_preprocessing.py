from pathlib import Path

import pandas as pd

# مسیر پایه پروژه
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = BASE_DIR / "data" / "raw"
PREPROCESSED_PATH = BASE_DIR / "data" / "preprocessed"


def load_data():
    """
    Load raw Rossmann dataset
    """
    train = pd.read_csv(RAW_PATH / "train.csv", dtype={"StateHoliday": str})
    store = pd.read_csv(RAW_PATH / "store.csv")

    return train, store


def merge_data(train, store):
    """
    Merge train and store datasets
    """
    df = pd.merge(train, store, on="Store", how="left")
    return df


def create_date_features(df):
    """
    Extract useful date features
    """
    df["Date"] = pd.to_datetime(df["Date"])

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["DayOfWeek"] = df["Date"].dt.dayofweek

    df.drop(columns=["Date"], inplace=True)

    return df


def handle_missing_values(df):
    """
    Handle missing values intelligently
    """
    df["CompetitionDistance"] = df["CompetitionDistance"].fillna(
        df["CompetitionDistance"].median()
    )

    df["CompetitionOpenSinceMonth"] = df["CompetitionOpenSinceMonth"].fillna(0)
    df["CompetitionOpenSinceYear"] = df["CompetitionOpenSinceYear"].fillna(0)

    df["Promo2SinceWeek"] = df["Promo2SinceWeek"].fillna(0)
    df["Promo2SinceYear"] = df["Promo2SinceYear"].fillna(0)

    df["PromoInterval"] = df["PromoInterval"].fillna("None")

    return df


def filter_open_stores(df):
    """
    Keep only open stores
    """
    df = df[df["Open"] == 1]
    df.drop(columns=["Open"], inplace=True)

    return df


def encode_categorical(df):
    """
    One-Hot Encode categorical features
    """
    df["StateHoliday"] = df["StateHoliday"].astype(str).replace("0", "None")
    df["IsStateHoliday"] = (df["StateHoliday"] != "None").astype(int)

    categorical_columns = [
        "StateHoliday",
        "StoreType",
        "Assortment",
        "PromoInterval",
    ]

    df = pd.get_dummies(
        df,
        columns=categorical_columns,
        drop_first=False,
        dtype=int,
    )

    return df


def save_preprocessed(df):
    """
    Save processed dataset
    """
    PREPROCESSED_PATH.mkdir(parents=True, exist_ok=True)
    df.to_csv(PREPROCESSED_PATH / "preprocessed.csv", index=False)


def clean_data():
    """
    Full preprocessing pipeline
    """
    train, store = load_data()
    df = merge_data(train, store)
    df = create_date_features(df)
    df = handle_missing_values(df)
    df = filter_open_stores(df)
    df = encode_categorical(df)

    save_preprocessed(df)

    return df


if __name__ == "__main__":
    df = clean_data()
    print("Preprocessing completed ✅")
    print(f"Final shape: {df.shape}")

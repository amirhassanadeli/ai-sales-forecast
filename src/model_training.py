from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# مسیر پایه
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "preprocessed" / "preprocessed.csv"
MODEL_PATH = BASE_DIR / "models"


def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


def time_based_split(df, split_ratio=0.8):
    """
    Split data chronologically (important for time series)
    """
    df = df.sort_values("Year")

    split_index = int(len(df) * split_ratio)

    train = df.iloc[:split_index]
    test = df.iloc[split_index:]

    X_train = train.drop("Sales", axis=1)
    y_train = train["Sales"]

    X_test = test.drop("Sales", axis=1)
    y_test = test["Sales"]

    return X_train, X_test, y_train, y_test


def build_pipeline():
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("models", LinearRegression())
    ])
    return pipeline


def evaluate_model(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("Model Evaluation 📊")
    print("-------------------")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R2   : {r2:.4f}")


def save_model(pipeline):
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH / "model.pkl")


def train():
    df = load_data()

    X_train, X_test, y_train, y_test = time_based_split(df)

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    evaluate_model(y_test, y_pred)
    save_model(pipeline)

    print("\nModel saved successfully ✅")


if __name__ == "__main__":
    train()

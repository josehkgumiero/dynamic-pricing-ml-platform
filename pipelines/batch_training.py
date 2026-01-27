import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from feature_store.build_features import build_features
from models.gbm_model import build_model

def train():
    df = pd.read_csv("data/olist_order_items_dataset.csv")

    df["target_price"] = df["price"] * (
        1 + (df["freight_value"] > df["freight_value"].median()) * 0.05
    )

    X = build_features(df)
    y = df["target_price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = build_model()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    print(f"MSE: {mse:.4f}")

    joblib.dump(model, "model.joblib")

if __name__ == "__main__":
    train()
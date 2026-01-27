import pandas as pd

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["total_value"] = df["price"] + df["freight_value"]
    df["high_freight"] = (df["freight_value"] > df["freight_value"].median()).astype(int)

    return df[[
        "price",
        "freight_value",
        "total_value",
        "high_freight"
    ]]
import pandas as pd
import os
from encoding.encoding import encode_categorical
from modelpre.scalestd import calculate_and_save_params, apply_standard_scale

def load_feature_names(path="data/feature_names.txt"):
    if not os.path.exists(path): return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def get_processed_data(input_csv, mode="train"):
    all_cols = load_feature_names()
    df = pd.read_csv(input_csv)

    if mode == "train":
        df = df[all_cols].dropna()
        df = encode_categorical(df)
        
        features_only = all_cols[:-1]
        target_col = all_cols[-1]

        # CHỈ scale X, KHÔNG scale y
        calculate_and_save_params(df, features_only)
        X_scaled_df = apply_standard_scale(df, features_only)
        
        # Kết hợp lại với y nguyên bản
        df_final = X_scaled_df.copy()
        df_final[target_col] = df[target_col].values

        if not os.path.exists("data"): os.makedirs("data")
        df_final.to_csv("data/data_scaled.csv", index=False)

        X = X_scaled_df.values
        y = df[target_col].values.reshape(-1, 1)
        return X, y

    elif mode == "predict":
        features_only = all_cols[:-1]
        missing = [col for col in features_only if col not in df.columns]
        if missing: raise ValueError(f"Thiếu cột: {missing}")

        df = df[features_only]
        df = encode_categorical(df)
        df_scaled = apply_standard_scale(df, features_only)
        return df_scaled.values, df_scaled
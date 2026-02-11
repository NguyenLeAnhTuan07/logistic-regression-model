import pandas as pd
import numpy as np
from encoding.encoding import encode_categorical

def load_feature_names(path="data/feature_names.txt"):
    with open(path, "r") as f:
        features = [line.strip() for line in f if line.strip()]
    return features


def preprocess(
    csv_path,
    feature_path="data/feature_names.txt",
    mean=None,
    std=None,
    training=True
):
    df = pd.read_csv(csv_path)

    # 1. Đọc feature names
    feature_names = load_feature_names(feature_path)

    # 2. Label là cột cuối
    label_col = feature_names[-1]
    feature_cols = feature_names[:-1]

    # 3. TRAIN vs PREDICT
    if training:
        # train: cần cả feature + label
        df = df[feature_names]
    else:
        # predict: chỉ cần feature
        df = df[feature_cols]

    # 4. Drop missing
    df = df.dropna()

    # 5. Encode categorical
    df = encode_categorical(df)

    # 6. Lấy X
    X = df[feature_cols].values.astype(float)

    # 7. Scale
    if training:
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1
    else:
        if mean is None or std is None:
            raise ValueError("Predict cần mean & std từ train")

    X = (X - mean) / std

    # 8. Lấy y nếu là training
    if training:
        y = df[label_col].values.reshape(-1, 1).astype(float)
        return X, y, feature_cols, mean, std
    else:
        return X, feature_cols

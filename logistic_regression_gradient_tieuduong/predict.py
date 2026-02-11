import numpy as np
import pandas as pd
from modelpre.preprocessing import preprocess
from modelpre.model import predict

def load_wb(path="wb/wb.csv"):
    df = pd.read_csv(path)
    w = df.iloc[0, :-1].values.reshape(-1, 1)
    b = df.iloc[0, -1]
    return w, b

def main():
    # load model
    w, b = load_wb()

    # load scaler
    mean = np.load("mest/mean.npy")
    std = np.load("mest/std.npy")

    # preprocess predict data
    X_new, feature_names = preprocess(
        "dudoan.csv",
        mean=mean,
        std=std,
        training=False
    )

    y_pred = predict(X_new, w, b)

    print("du doan tieu duong(1 (co) / 0 (khong):")
    print(y_pred)

if __name__ == "__main__":
    main()

import pandas as pd
import os
from modelpre.preprocessing import get_processed_data, load_feature_names
from modelpre.model import fit

def main():
    # 1. X đã được scale, y giữ nguyên 0/1
    X_train, y_train = get_processed_data("data/data.csv", mode="train")

    print(f"Huấn luyện Logistic với {X_train.shape[0]} mẫu")

    # 2. Huấn luyện với hàm fit (Logistic)
    w, b = fit(X_train, y_train)

    # 3. Lưu trọng số
    feat_names = load_feature_names()[:-1]
    wb_df = pd.DataFrame([list(w.flatten()) + [b]], columns=feat_names + ["bias"])

    if not os.path.exists("meta"): os.makedirs("meta")
    wb_df.to_csv("meta/wb.csv", index=False)
    print("Mô hình Logistic đã được lưu tại meta/wb.csv")

if __name__ == "__main__":
    main()
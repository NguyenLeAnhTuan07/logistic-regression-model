import os
import numpy as np
import pandas as pd
from modelpre.preprocessing import preprocess
from modelpre.model import fit


def save_wb_to_csv(w, b, feature_names, meta_dir, filename="wb.csv"):
    # Chuyển w sang numpy array và làm phẳng thành vector 1 chiều
    w = np.array(w).reshape(-1)

    # Tạo danh sách tên cột: các feature + bias
    columns = feature_names + ["bias"]

    # Ghép giá trị weight và bias thành một dòng dữ liệu
    values = list(w) + [b]

    # Tạo DataFrame gồm 1 dòng (weights + bias)
    df = pd.DataFrame([values], columns=columns)

    # Tạo đường dẫn lưu file CSV trong thư mục meta_dir
    path = os.path.join(meta_dir, "wb.csv")

    # Ghi DataFrame ra file CSV, không lưu index
    df.to_csv(path, index=False)
    print(f"Saved weights & bias to {path}")



def main():
    # Tiền xử lý dữ liệu
    # X: ma trận đặc trưng
    # y: nhãn
    # feature_names: tên các feature
    # mean, std: giá trị dùng để chuẩn hóa dữ liệu
    X, y, feature_names, mean, std = preprocess(
        "data/data.csv",   # Đường dẫn file dữ liệu
        training=True      # Đang ở chế độ huấn luyện
    )
    # Huấn luyện mô hình
    w, b = fit(X, y)

    print("Weights:\n", w)
    print("Bias:", b)
    
    os.makedirs("wb", exist_ok=True)
    save_wb_to_csv(w, b, feature_names , "wb")

    os.makedirs("mest", exist_ok=True)
    np.save("mest/mean.npy", mean)
    np.save("mest/std.npy", std)



if __name__ == "__main__":
    main()

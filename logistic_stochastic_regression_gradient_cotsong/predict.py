import pandas as pd
import os
import numpy as np
from modelpre.preprocessing import get_processed_data
from modelpre.model import predict

def main():
    # 1. Đường dẫn file đầu vào
    input_file = "predict/dudoan.csv"

    if not os.path.exists(input_file):
        print(f"Lỗi: Không tìm thấy file {input_file}. Hãy tạo file và nhập dữ liệu cần dự đoán.")
        return

    # 2. TIỀN XỬ LÝ & SCALE: 
    # Hàm này sẽ thực hiện Encoding + Standard Scale dựa trên params từ tập Train
    try:
        # X_test là numpy array để dự đoán, df_scaled là DataFrame chứa dữ liệu đã scale
        X_test, df_scaled = get_processed_data(input_file, mode="predict")
    except Exception as e:
        print(f"Lỗi khi tiền xử lý dữ liệu: {e}")
        return

    # 3. LƯU DỮ LIỆU ĐÃ SCALE: (Yêu cầu của bạn)
    # Lưu file dudoan_scaled.csv để bạn có thể kiểm tra các giá trị sau khi chuẩn hóa
    if not os.path.exists("predict"):
        os.makedirs("predict")
    
    scaled_output_path = "predict/dudoan_scaled.csv"
    df_scaled.to_csv(scaled_output_path, index=False)
    print(f"--- Đã lưu dữ liệu đã scale tại: {scaled_output_path} ---")

    # 4. LOAD TRỌNG SỐ: Đọc w và b từ meta/wb.csv
    wb_path = "meta/wb.csv"
    if not os.path.exists(wb_path):
        print("Lỗi: Không tìm thấy meta/wb.csv. Bạn cần chạy train.py trước!")
        return

    df_wb = pd.read_csv(wb_path)
    # Lấy các cột trừ cột cuối cùng (bias) làm w
    w = df_wb.iloc[0, :-1].values.reshape(-1, 1)
    # Cột cuối cùng là b
    b = df_wb.iloc[0, -1]

    # 5. DỰ ĐOÁN: Kết quả là xác suất (0 đến 1) nhờ hàm Sigmoid
    y_prob = predict(X_test, w, b)

    # 6. PHÂN LOẠI: Chuyển xác suất sang nhãn 0/1 (Ngưỡng 0.5)
    y_pred = (y_prob >= 0.5).astype(int)

    # 7. XUẤT KẾT QUẢ RA MÀN HÌNH
    print("\n" + "="*40)
    print("KẾT QUẢ DỰ ĐOÁN")
    print("="*40)
    for i, (prob, label) in enumerate(zip(y_prob.flatten(), y_pred.flatten())):
        trang_thai = "(1)" if label == 1 else "(0)"
        print(f"Mẫu số {i+1:<3}: {trang_thai:<15} | Xác suất: {prob:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()
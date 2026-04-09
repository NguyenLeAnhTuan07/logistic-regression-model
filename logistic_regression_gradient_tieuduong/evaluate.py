import pandas as pd
import numpy as np
import os
from modelpre.model import fit, predict
from modelpre.preprocessing import load_feature_names

def calculate_classification_metrics(y_true, y_prob):
    """Tính toán Accuracy, Precision, Recall, F1, AUC"""
    y_true = np.array(y_true).flatten()
    y_prob = np.array(y_prob).flatten()
    y_pred = (y_prob >= 0.5).astype(int)
    
    eps = 1e-8
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)

    # Tính AUC (Diện tích dưới đường cong ROC)
    desc_score_indices = np.argsort(y_prob)[::-1]
    y_true_sorted = y_true[desc_score_indices]
    num_pos = np.sum(y_true == 1)
    num_neg = np.sum(y_true == 0)
    
    if num_pos == 0 or num_neg == 0:
        auc = 0.5
    else:
        tp_cur, fp_cur, tp_prev, fp_prev, auc = 0, 0, 0, 0, 0
        for i in range(len(y_true_sorted)):
            if y_true_sorted[i] == 1: tp_cur += 1
            else:
                fp_cur += 1
                auc += (fp_cur - fp_prev) * (tp_cur + tp_prev) / 2
                fp_prev, tp_prev = fp_cur, tp_cur
        auc = auc / (num_pos * num_neg)

    return accuracy, precision, recall, f1, auc, y_pred

def get_tfpn_label(y_true, y_pred):
    """Gán nhãn TP, TN, FP, FN cho từng mẫu dữ liệu"""
    results = []
    for t, p in zip(y_true, y_pred):
        if t == 1 and p == 1: results.append("TP")
        elif t == 0 and p == 0: results.append("TN")
        elif t == 1 and p == 0: results.append("FN")
        else: results.append("FP")
    return results

def run_k_fold_cv(X, y, k=5):
    m = X.shape[0]
    indices = np.arange(m)
    np.random.shuffle(indices)

    fold_sizes = np.full(k, m // k)
    fold_sizes[: m % k] += 1
    current = 0
    folds = [indices[current:current + (current := current + size)] for size in fold_sizes]

    history = {"Accuracy": [], "Precision": [], "Recall": [], "F1": [], "AUC": []}
    all_tfpn_details = []

    print(f"\n--- Đang thực hiện {k}-Fold Cross Validation (Logistic) ---")
    print(f"{'Fold':<8} | {'Acc':<8} | {'Prec':<8} | {'Recall':<8} | {'F1':<8} | {'AUC':<8}")
    print("-" * 65)

    for i in range(k):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        w_f, b_f = fit(X_train, y_train)
        y_prob_f = predict(X_test, w_f, b_f)
        acc, prec, rec, f1, auc, y_pred_f = calculate_classification_metrics(y_test, y_prob_f)
        
        # Lưu TFPN cho từng mẫu
        tfpn_labels = get_tfpn_label(y_test.flatten(), y_pred_f.flatten())
        for idx_in_fold, label in enumerate(tfpn_labels):
            all_tfpn_details.append({"stt": test_idx[idx_in_fold], "TFPN": label})

        history["Accuracy"].append(acc)
        history["Precision"].append(prec)
        history["Recall"].append(rec)
        history["F1"].append(f1)
        history["AUC"].append(auc)
        print(f"Fold {i+1:<3} | {acc:<8.4f} | {prec:<8.4f} | {rec:<8.4f} | {f1:<8.4f} | {auc:<8.4f}")

    # 1. Lưu file CSV chi tiết
    df_tfpn = pd.DataFrame(all_tfpn_details).sort_values(by="stt")
    if not os.path.exists("predict"): os.makedirs("predict")
    df_tfpn.to_csv("predict/tfpn_details.csv", index=False)

    # 2. In thống kê số lượng TP, TN, FP, FN
    counts = df_tfpn['TFPN'].value_counts()
    print("\n" + "=" * 30)
    print("THỐNG KÊ CHI TIẾT TFPN")
    print("=" * 30)
    for label in ["TP", "TN", "FP", "FN"]:
        print(f"  {label:<4}: {counts.get(label, 0):>5} mẫu")
    print("-" * 30)
    print(f"Tổng cộng: {len(df_tfpn)} mẫu")
    print(f"File: predict/tfpn_details.csv")
    print("=" * 30)

    return history

def main():
    path = "data/data_scaled.csv"
    if not os.path.exists(path):
        print("Lỗi: Không tìm thấy data_scaled.csv. Hãy chạy train.py trước!")
        return

    df = pd.read_csv(path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1:].values 

    cv_history = run_k_fold_cv(X, y, k=5)

    # --- PHẦN 1: ĐÁNH GIÁ TỔNG QUÁT ---
    print("\n" + "=" * 58)
    print("ĐÁNH GIÁ TỔNG QUÁT (K-FOLD MEAN)")
    print("=" * 58)
    for metric in ["Accuracy", "Precision", "Recall", "F1", "AUC"]:
        mean_val = np.mean(cv_history[metric])
        print(f"{metric:<15}: {mean_val:.4f} ({mean_val*100:.2f}%)")
    print("-" * 58)

    # --- PHẦN 2: CV SCORE (Lựa chọn Mi) ---
    print("\n" + "=" * 58)
    print("CV SCORE")
    print("=" * 58)
    print("Bạn muốn tính CV Score dựa trên chỉ số nào (Mi)?")
    options = {
        "1": "Accuracy",
        "2": "Precision",
        "3": "Recall",
        "4": "F1",
        "5": "AUC"
    }
    for k, v in options.items():
        print(f"  {k}. {v}")
    
    choice = input("Nhập lựa chọn (1-5): ").strip()

    if choice not in options:
        print("Lựa chọn không hợp lệ. Kết thúc chương trình.")
        return

    chosen_metric = options[choice]
    scores = np.array(cv_history[chosen_metric])
    cv_mean = np.mean(scores)
    cv_std = np.std(scores)

    print("\n" + "-" * 58)
    print(f"KẾT QUẢ CV SCORE (Mi = {chosen_metric})")
    print(f"  CV_Score_mean  :  {cv_mean:.6f}")
    print(f"  CV_Score_std   :  {cv_std:.6f}")
    print("=" * 58)

if __name__ == "__main__":
    main()
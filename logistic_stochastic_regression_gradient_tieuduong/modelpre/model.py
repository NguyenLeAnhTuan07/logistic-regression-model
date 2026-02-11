import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def initialize_parameters(n):
    """
    Khởi tạo trọng số và bias
    n: số feature
    """
    w = np.zeros((n, 1))
    b = 0.0
    return w, b


#logistic
def predict(X, w, b):
    z = X @ w + b
    return sigmoid(z)

def compute_loss(y, y_hat):
    eps = 1e-8  # tránh log(0)
    return -np.mean(
        y * np.log(y_hat + eps) +
        (1 - y) * np.log(1 - y_hat + eps)
    )

def compute_gradients(X, y, y_hat):
    m = X.shape[0]
    dw = (1 / m) * X.T @ (y_hat - y)
    db = (1 / m) * np.sum(y_hat - y)
    return dw, db



def fit(X, y, lr=0.01, epochs=10000, patience=1000):
    m, n = X.shape
    w, b = initialize_parameters(n)

    best_w = w.copy()
    best_b = b
    best_loss = float("inf")
    wait = 0

    for epoch in range(epochs):

        # Shuffle dữ liệu mỗi epoch
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        # ====== SGD: cập nhật từng mẫu ======
        for i in range(m):
            xi = X_shuffled[i:i+1]     # shape (1, n)
            yi = y_shuffled[i:i+1]     # shape (1, 1)

            y_hat = predict(xi, w, b)

            dw, db = compute_gradients(xi, yi, y_hat)

            w -= lr * dw
            b -= lr * db

        # ====== Tính loss toàn bộ dataset để theo dõi ======
        y_hat_full = predict(X, w, b)
        loss = compute_loss(y, y_hat_full)

        if loss < best_loss:
            best_loss = loss
            best_w = w.copy()
            best_b = b
            wait = 0
        else:
            wait += 1

        if wait >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    return best_w, best_b







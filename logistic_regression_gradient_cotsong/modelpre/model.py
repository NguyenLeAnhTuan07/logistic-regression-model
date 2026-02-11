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

"""
#linear
def predict(X, w, b):
    
    #Dự đoán y_hat = Xw + b
    
    return X @ w + b

def compute_loss(y, y_hat):
    
    #Hàm mất mát MSE
    
    m = y.shape[0]
    loss = (1 / m) * np.sum((y_hat - y) ** 2)
    return loss

def compute_gradients(X, y, y_hat):
    
    #Tính gradient cho w và b
    
    m = X.shape[0]

    dw = (2 / m) * X.T @ (y_hat - y)
    db = (2 / m) * np.sum(y_hat - y)

    return dw, db
"""
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



def fit(X, y, lr=0.2, epochs=50000, patience=200):
    m, n = X.shape
    w, b = initialize_parameters(n)

    best_w = w.copy()
    best_b = b
    best_loss = float("inf")
    wait = 0

    for epoch in range(epochs):
        y_hat = predict(X, w, b)
        loss = compute_loss(y, y_hat)

        dw, db = compute_gradients(X, y, y_hat)
        w -= lr * dw
        b -= lr * db

        # tính lại sau khi update
        y_hat_new = predict(X, w, b)
        loss_new = compute_loss(y, y_hat_new)

        if loss_new < best_loss:
            best_loss = loss_new
            best_w = w.copy()
            best_b = b
            wait = 0
        else:
            wait += 1

        if wait >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    return best_w, best_b






import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    n1, n2 = X.shape
    w = np.zeros(n2)
    b = 0

    for _ in range(steps):
        z = b + np.dot(X,w)
        p = _sigmoid(z)
        dw = np.dot(X.T , (p - y)) / n2
        db = sum(p - y) / n2

        w -= lr * dw
        b -= lr * db
        
    return (w, b)
        
    
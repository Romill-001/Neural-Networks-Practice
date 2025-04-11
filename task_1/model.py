import numpy as np

def model(X, y, w, b, cf):

    lrw= cf["learning_rate_w"]
    lrb = cf["learning_rate_b"]
    lambda_ = cf["lambda"]
    epochs = cf["num_epochs"]

    for _ in range(epochs):
        y_prediction = np.dot(X, w) + b
        error = y_prediction - y
        w_grad = (2 / len(X)) * np.dot(X.T, error) + 2 * lambda_ * w #L2 regularization
        b_grad = (2 / len(X)) * np.sum(error)
        w -= lrw * w_grad
        b -= lrb * b_grad
    
    return w, b
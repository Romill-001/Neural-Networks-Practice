import numpy as np

def accuracy(y, y_pred):
    c = 0
    for i in range(len(y)):
        if y[i] == y_pred[i]:
            c += 1

    return c / (len(y) - 1)

def precision(y, y_pred):
    TP, FP = 0, 0
    for i in range(len(y)):
        if y[i] == 1 and y_pred[i] == 1:
            TP += 1
        elif y[i] == 1 and y_pred[i] == 0:
            FP += 1
    
    return TP / (TP + FP)

def recall(y, y_pred):
    TP, FN = 0, 0
    for i in range(len(y)):
        if y[i] == 1 and y_pred[i] == 1:
            TP += 1
        elif y[i] == 0 and y_pred[i] == 1:
            FN += 1
    
    return TP / (TP + FN)

def avg_harmonical(y, y_pred):
    p = precision(y, y_pred)
    r = recall(y, y_pred)

    return (2 * p * r) / (p + r)

def z_score_normalizer(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    return (X - mu) / (sigma + 1e-8)

def mse_metric(y_true, y_predicted):
    n = len(y_true)
    sum = 0
    for i in range(n):
        sum += (y_true[i] - y_predicted[i])**2
    return (1 / n) * sum

def rmse_metric(y_true, y_predicted):
    n = len(y_true)
    sum = 0
    for i in range(n):
        sum += (y_true[i] - y_predicted[i])**2
    return np.sqrt((1 / n) * sum)

def mae_metric(y_true, y_predicted):
    n = len(y_true)
    sum = 0
    for i in range(n):
        sum += np.abs(y_true[i] - y_predicted[i])
    return (1 / n) * sum

def mape_metric(y_true, y_predicted):
    return np.mean(np.abs((y_true - y_predicted) / y_true)) * 100

import numpy as np
import pandas as pd

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

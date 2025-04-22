import matplotlib.pyplot as plt

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
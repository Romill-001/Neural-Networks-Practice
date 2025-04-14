import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def polinom(x : float, w : np.array):
    """
    Строим полином
    """
    return sum(w[i] * x**i for i in range(len(w)))

def create_design_matrix(x : np.array, degree : int):
    """
    создаём матрицу плана вида:\n
    |1 (x_1) ... (x_1)^d|\n
    |1 (x_2) ... (x_2)^d|\n
    |1 (x_3) ... (x_3)^d|\n
    |  ...   ...   ...  |\n
    |1 (x_n) ... (x_n)^d|\n
    """
    mtr = np.zeros((len(x), degree + 1))
    for i in range(degree + 1):
        mtr[:, i] = x ** i
    return mtr

def regression_solvation(x : np.array, y : np.array):
    """
    **Метод наименьших квадратов**
    y = Xw, w = (w_0, w_1, ..., w_n)\n
    Сумма квадратов ошибок: S = ||y - Xw||^2\n
    Находим градиент: dS/dw = -2X^T(y - Xw) = 0\n
    X^Ty = X^TXw -> w = (X^TX)^(-1)X^Ty\n
    """
    w = np.linalg.solve(x.T @ x, x.T @ y)
    return w

def main():
    x = np.arange(1, 10, 0.1)
    noise = np.random.normal(0, 0.05, len(x))
    y = np.log(x)
    y += noise

    degree = 5
    design_matrix = create_design_matrix(x, degree)
    w = regression_solvation(design_matrix, y)

    x_new = np.arange(1, 10, 0.01)
    y_pred = np.array([polinom(x, w) for x in x_new])
    
    plt.plot(x_new, np.log(x_new), '-.')
    plt.plot(x_new, y_pred)
    plt.show()


if __name__ == '__main__':
    main()
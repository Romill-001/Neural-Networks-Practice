import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import *

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
    noise = np.random.normal(0, 0.01, len(x))
    y = np.log(x)
    y += noise
    
    x_train, y_train = x[:69], y[:69]
    x_test, y_test = x[70:], y[70:]
    degree = 5
    design_matrix = create_design_matrix(x_train, degree)
    w = regression_solvation(design_matrix, y_train)

    
    y_pred = np.array([polinom(x, w) for x in x_test])
    
    # arr_train = []
    # arr_test = []
    # for deg in range(1, 11):
    #     design_matrix = create_design_matrix(x_train, deg)
    #     w = regression_solvation(design_matrix, y_train)
    #     y_pred_train = np.array([polinom(x, w) for x in x_train])
    #     arr_train.append(mse_metric(y_train, y_pred_train))
    #     y_pred_test = np.array([polinom(x, w) for x in x_test])
    #     arr_test.append(mse_metric(y_test, y_pred_test))
    

    # plt.plot(np.arange(1, 11, 1), arr_train, '-r', label="Ошибка трейна")
    # plt.plot(np.arange(1, 11, 1), arr_test, '-b', label="Ошибка тестирования")
    # plt.grid()
    # plt.legend()
    # plt.xlim((0, 6))
    # plt.ylim((0,0.5))
    # plt.show()


    # fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    # for i, deg in enumerate(range(5, 11)):
    #     ax = axes.flat[i]
    #     design_matrix = create_design_matrix(x_train, deg)
    #     w = regression_solvation(design_matrix, y_train)
    #     y_pred = np.array([polinom(x, w) for x in x_test])
    #     ax.plot(x_test, y_test, '-r', label='Истинные значения')
    #     ax.plot(x_test, y_pred, label='Предсказанные значения для степени {}'.format(deg))
    #     ax.set_title(f'Полином степени {deg}')
    #     ax.grid()
    #     ax.legend()

    plt.plot(x_test, y_test, '-r', label='Истинные значения')
    plt.plot(x_test, y_pred, label='Предсказанные значения для степени {}'.format(degree))
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
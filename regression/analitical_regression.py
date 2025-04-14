import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import *

def main():
    train_data = pd.read_csv("./train.csv")
    y = z_score_normalizer(np.array(train_data["SalePrice"]))
    X = train_data.drop(["Id", "SalePrice"], axis=1)
    X = np.array(X.select_dtypes(include=['float64', 'int64']).fillna(0))
    X = z_score_normalizer(X)
    #добавляем фиктивный признак, чтобы учитывать смещение b
    X = np.hstack((np.ones((X.shape[0], 1)), X))

    #
    w = np.linalg.inv(X.T @ X) @ X.T @ y

    y_p = X @ w

    print(f"MSE {mse_metric(y, y_p)}")
    print(f"RMSE {rmse_metric(y, y_p)}")
    print(f"MAE {mae_metric(y, y_p)}")
    print(f"MAPE {mape_metric(y, y_p)}")
    plt.scatter(y, y_p, 2)
    plt.plot(y_p, y_p, "r")
    plt.show()


if __name__ == '__main__':
    main()
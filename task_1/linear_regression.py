from utils import *
import matplotlib.pyplot as plt
from model import model

cf = {
    "learning_rate_w" : 0.000001,
    "learning_rate_b" : 0.000001,
    "lambda" : 0.01,
    "num_epochs" : 1000
}


train_df = pd.read_csv("./train.csv")
test_df = pd.read_csv("./test.csv")


y = z_score_normalizer(np.array(train_df["SalePrice"]))
X = train_df.drop(["Id", "SalePrice"], axis=1)
X = X.select_dtypes(include=['float64', 'int64']).fillna(0)
X = z_score_normalizer(X)

X = np.array(X)
y = np.array(y)

w = np.zeros(X.shape[1])
b = np.mean(y)

w, b = model(X, y, w, b, cf)

y_p = z_score_normalizer(np.dot(X, w) + b)

print(f"MSE {mse_metric(y, y_p)}")
print(f"RMSE {rmse_metric(y, y_p)}")
print(f"MAE {mae_metric(y, y_p)}")
print(f"MAPE {mape_metric(y, y_p)}")
print(f"Веса w: {w}, смещение b {b}")


plt.scatter(np.arange(len(X)), y)
plt.plot(np.arange(len(X)), y_p, "-r")

plt.show()

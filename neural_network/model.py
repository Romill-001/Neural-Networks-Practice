from torchvision.datasets import MNIST
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from utils import *
import numpy as np


class NeuralNetwork:
    def __init__(self, layer_sizes, task='regression', init_method='xavier', lr=0.01, activation = 'relu'):
        """
        :param layer_sizes: list of int, например [input_size, hidden1, ..., output_size]
        :param task: 'regression', 'binary_classification', 'multiclass_classification'
        :param init_method: 'xavier' или 'he' (для ReLU)
        :param lr: скорость обучения
        """
        self.layer_sizes = layer_sizes
        self.L = len(layer_sizes)
        self.lr = lr
        self.task = task
        self.parameters = {}
        self.cache = {}
        self.activation = activation

        for i in range(1, self.L):
            if init_method == 'xavier':
                scale = np.sqrt(1. / layer_sizes[i - 1])
            elif init_method == 'he':
                scale = np.sqrt(2. / layer_sizes[i - 1])
            else:
                raise ValueError("init_method должен быть 'xavier' или 'he'")
            self.parameters[f'W{i}'] = np.random.normal(0, scale, size=(layer_sizes[i-1], layer_sizes[i]))
            self.parameters[f'b{i}'] = np.zeros((1, layer_sizes[i]))

    def _activate(self, x):
        if self.activation == 'relu':
            return self._relu(x)
        elif self.activation == 'tanh':
            return np.tanh(x)
        else:
            raise ValueError("Неизвестная активация")
        
    def _activate_derivative(self, x):
        if self.activation == 'relu':
            return self._relu_derivative(x)
        elif self.activation == 'tanh':
            return 1 - np.tanh(x) ** 2
        else:
            raise ValueError("Неизвестная активация")
        
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_derivative(self, x):
        return x * (1 - x)

    def _relu(self, x):
        return np.maximum(0, x)

    def _relu_derivative(self, x):
        return (x > 0).astype(float)

    def _softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def _mse_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def _cross_entropy_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        return loss

    def _forward(self, X):
        self.cache['A0'] = X
        A_prev = X

        for i in range(1, self.L - 1):
            W = self.parameters[f'W{i}']
            b = self.parameters[f'b{i}']
            Z = np.dot(A_prev, W) + b
            A = self._activate(Z)
            self.cache[f'Z{i}'] = Z
            self.cache[f'A{i}'] = A
            A_prev = A

        # Выходной слой
        W_out = self.parameters[f'W{self.L - 1}']
        b_out = self.parameters[f'b{self.L - 1}']
        Z_out = np.dot(A_prev, W_out) + b_out

        if self.task in ['binary_classification']:
            A_out = self._sigmoid(Z_out)
        elif self.task in ['multiclass_classification']:
            A_out = self._softmax(Z_out)
        else:
            A_out = Z_out  # регрессия

        self.cache[f'A{self.L - 1}'] = A_out
        self.cache[f'Z{self.L - 1}'] = Z_out
        return A_out

    def _backward(self, y):
        m = y.shape[0]
        grads = {}

        if self.task == 'binary_classification':
            dZ = (self.cache[f'A{self.L - 1}'] - y)
        elif self.task == 'multiclass_classification':
            dZ = (self.cache[f'A{self.L - 1}'] - y)
        else:
            dZ = (self.cache[f'Z{self.L - 1}'] - y)

        A_prev = self.cache[f'A{self.L - 2}']
        grads[f'dW{self.L - 1}'] = np.dot(A_prev.T, dZ) / m
        grads[f'db{self.L - 1}'] = np.sum(dZ, axis=0, keepdims=True) / m

        dA_prev = np.dot(dZ, self.parameters[f'W{self.L - 1}'].T)

        for i in reversed(range(1, self.L - 1)):
            Z = self.cache[f'Z{i}']
            A_prev = self.cache[f'A{i - 1}']
            dZ = dA_prev * self._activate_derivative(Z)
            grads[f'dW{i}'] = np.dot(A_prev.T, dZ) / m
            grads[f'db{i}'] = np.sum(dZ, axis=0, keepdims=True) / m
            dA_prev = np.dot(dZ, self.parameters[f'W{i}'].T)

        for i in range(1, self.L):
            self.parameters[f'W{i}'] -= self.lr * grads[f'dW{i}']
            self.parameters[f'b{i}'] -= self.lr * grads[f'db{i}']

    def train(self, X_train, y_train, epochs=10, batch_size=32):
        num_samples = X_train.shape[0]
        indices = np.arange(num_samples)

        for epoch in range(epochs):
            np.random.shuffle(indices)
            epoch_loss = 0.0
            correct_preds = 0

            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                batch_indices = indices[start:end]
                X_batch = X_train[batch_indices]
                y_batch = y_train[batch_indices]

                output = self._forward(X_batch)

                if self.task in ['binary_classification', 'multiclass_classification']:
                    loss = self._cross_entropy_loss(y_batch, output)
                    preds = np.argmax(output, axis=1)
                    true_labels = np.argmax(y_batch, axis=1)
                    correct_preds += np.sum(preds == true_labels)
                else:
                    loss = self._mse_loss(y_batch, output)

                epoch_loss += loss * (end - start)
                self._backward(y_batch)

            avg_loss = epoch_loss / num_samples
            if self.task in ['binary_classification', 'multiclass_classification']:
                acc = correct_preds / num_samples
                print(f"Epoch {epoch}, Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")
            else:
                print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")

    def predict(self, X):
        output = self._forward(X)
        if self.task in ['binary_classification']:
            return (output > 0.5).astype(int)
        elif self.task in ['multiclass_classification']:
            return np.argmax(output, axis=1)
        else:
            return output
        
class LinearDataset(Dataset):
    def __init__(self, n_samples=1000, noise=0.1):
        self.x = np.random.rand(n_samples, 1) * 10
        self.y = self.x + np.random.normal(0, noise, size=(n_samples, 1))
        self.x = self.x.astype(np.float32)
        self.y = self.y.astype(np.float32)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class LinearModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 1)
        )
        self.loss_fn = torch.nn.MSELoss()
    
    def forward(self, x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)
    
class SinDataset(Dataset):
    def __init__(self, n_samples=1000, noise=0.05):
        self.x = np.random.rand(n_samples, 1) * 10
        self.y = np.sin(self.x) + np.random.normal(0, noise, size=(n_samples, 1))
        self.x = self.x.astype(np.float32)
        self.y = self.y.astype(np.float32)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class SinModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )
        self.loss_fn = torch.nn.MSELoss()
    
    def forward(self, x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
    
class HousePricesDataset(Dataset):
    def __init__(self, train=True):

        train_df = pd.read_csv("./data/train_reg.csv")
        
        y = np.array(train_df["SalePrice"]).reshape(-1, 1)
        X = train_df.drop(["Id", "SalePrice"], axis=1)
        X = X.select_dtypes(include=['float64', 'int64']).fillna(0)
        self.feature_names = X.columns.tolist()

        self.X_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

        # X = self.X_scaler.fit_transform(X)
        # y = self.y_scaler.fit_transform(y)
        X = z_score_normalizer(np.array(X))
        y = z_score_normalizer(np.array(y))
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        
        if train:
            self.X = torch.tensor(X_train, dtype=torch.float32)
            self.y = torch.tensor(y_train, dtype=torch.float32)
        else:
            self.X = torch.tensor(X_test, dtype=torch.float32)
            self.y = torch.tensor(y_test, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    def get_feature_names(self):
        return self.feature_names
    
    def inverse_transform_y(self, y_tensor):
        if isinstance(y_tensor, torch.Tensor):
            y_tensor = y_tensor.cpu().numpy()

        y_tensor = y_tensor.reshape(-1, 1)

        return self.y_scaler.inverse_transform(y_tensor)

class HousePriceModel(pl.LightningModule):
    def __init__(self, input_size):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )
        self.loss_fn = torch.nn.MSELoss()
    
    def forward(self, x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        self.log('train_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        preds = torch.argmax(y_pred, dim=1)
        self.log('test_loss', loss)
        return {'predictions': preds, 'targets': y}

    def on_test_start(self):
        self.all_preds = []
        self.all_targets = []

    def on_test_batch_end(self, outputs, batch, batch_idx):
        preds = outputs["predictions"]
        targets = outputs["targets"]
        self.all_preds.append(preds)
        self.all_targets.append(targets)

    def on_test_epoch_end(self):
        all_preds = torch.cat(self.all_preds)
        all_targets = torch.cat(self.all_targets)

        self.test_predictions = all_preds
        self.test_targets = all_targets

    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
    
class TitanicDataset(Dataset):
    def __init__(self, train=True):
        data = pd.read_csv("./data/train.csv")

        data = data.drop(["PassengerId", "Name", "Age", "Ticket", "Fare", "Cabin"],axis=1)
        data = pd.get_dummies(data=data, columns=['Sex', 'Embarked'], prefix=['sex', 'embarked'])

        X = data.drop('Survived',axis=1)
        y = data['Survived']
        
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        
        if train:
            self.x = X_train.astype(np.float32)
            self.y = y_train.astype(np.int64)
        else:
            self.x = X_test.astype(np.float32)
            self.y = y_test.astype(np.int64)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class TitanicModel(pl.LightningModule):
    def __init__(self, input_size=6):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_size, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 2)
        )
        self.loss_fn = torch.nn.CrossEntropyLoss()
    
    def forward(self, x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        self.log('train_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        preds = torch.argmax(y_pred, dim=1)
        self.log('test_loss', loss)
        return {'predictions': preds, 'targets': y}
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

class MNISTDataset(Dataset):
    def __init__(self, train=True):
        self.mnist = MNIST(root='./data', train=train, download=True,
                          transform=transforms.ToTensor())
    
    def __len__(self):
        return len(self.mnist)
    
    def __getitem__(self, idx):
        x, y = self.mnist[idx]
        return x.view(-1), y

class MNISTModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(784, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10)
        )
        self.loss_fn = torch.nn.CrossEntropyLoss()
    
    def forward(self, x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        self.log('train_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        preds = torch.argmax(y_pred, dim=1)
        self.log('test_loss', loss)
        return {'predictions': preds, 'targets': y}
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
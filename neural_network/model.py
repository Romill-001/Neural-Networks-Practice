import numpy as np


class NeuralNetwork:
    def __init__(self, layer_sizes, task='regression', init_method='xavier', lr=0.01):
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

        for i in range(1, self.L):
            if init_method == 'xavier':
                scale = np.sqrt(1. / layer_sizes[i - 1])
            elif init_method == 'he':
                scale = np.sqrt(2. / layer_sizes[i - 1])
            else:
                raise ValueError("init_method должен быть 'xavier' или 'he'")
            self.parameters[f'W{i}'] = np.random.normal(0, scale, size=(layer_sizes[i-1], layer_sizes[i]))
            self.parameters[f'b{i}'] = np.zeros((1, layer_sizes[i]))

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

    def forward(self, X):
        self.cache['A0'] = X
        A_prev = X

        for i in range(1, self.L - 1):
            W = self.parameters[f'W{i}']
            b = self.parameters[f'b{i}']
            Z = np.dot(A_prev, W) + b
            A = self._relu(Z)
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

    def backward(self, y):
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

        # Обратное распространение по скрытым слоям
        for i in reversed(range(1, self.L - 1)):
            Z = self.cache[f'Z{i}']
            A_prev = self.cache[f'A{i - 1}']
            dZ = dA_prev * self._relu_derivative(Z)
            grads[f'dW{i}'] = np.dot(A_prev.T, dZ) / m
            grads[f'db{i}'] = np.sum(dZ, axis=0, keepdims=True) / m
            dA_prev = np.dot(dZ, self.parameters[f'W{i}'].T)

        # Обновление параметров
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

                output = self.forward(X_batch)

                if self.task in ['binary_classification', 'multiclass_classification']:
                    loss = self._cross_entropy_loss(y_batch, output)
                    preds = np.argmax(output, axis=1)
                    true_labels = np.argmax(y_batch, axis=1)
                    correct_preds += np.sum(preds == true_labels)
                else:
                    loss = self._mse_loss(y_batch, output)

                epoch_loss += loss * (end - start)
                self.backward(y_batch)

            avg_loss = epoch_loss / num_samples
            if self.task in ['binary_classification', 'multiclass_classification']:
                acc = correct_preds / num_samples
                print(f"Epoch {epoch}, Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")
            else:
                print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")

    def predict(self, X):
        output = self.forward(X)
        if self.task in ['binary_classification']:
            return (output > 0.5).astype(int)
        elif self.task in ['multiclass_classification']:
            return np.argmax(output, axis=1)
        else:
            return output
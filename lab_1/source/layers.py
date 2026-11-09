import numpy as np


class Linear:
    """Полносвязный слой: X (batch, in_features) → Y (batch, out_features)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rng: np.random.Generator,
    ):
        # Общий rng сохраняет порядок инициализации весов при сборке модели.
        self.W = rng.standard_normal((in_features, out_features)) * np.sqrt(
            1 / in_features
        )
        self.b = np.zeros((1, out_features))

    def forward(self, X):
        """Сохраняет X для расчёта dW в обратном проходе."""
        self.X = X
        return np.dot(X, self.W) + self.b

    def backward(self, dY):
        """Принимает dY (batch, out_features), возвращает dX (batch, in_features)."""
        self.dW = np.dot(self.X.T, dY)
        self.db = np.sum(dY, axis=0, keepdims=True)
        return np.dot(dY, self.W.T)

    def update_parameters(self, learning_rate):
        """Обновляет параметры после завершения обратного прохода."""
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db


class Sigmoid:
    """Поэлементная сигмоида; форма массива при проходах не меняется."""

    @staticmethod
    def activation(z):
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def activation_derivative(a):
        """Производная через уже вычисленную активацию a = sigmoid(z)."""
        return a * (1.0 - a)

    def forward(self, X):
        self.A = self.activation(X)
        return self.A

    def backward(self, dY):
        return dY * self.activation_derivative(self.A)


class Softmax:
    """Преобразует логиты (batch, classes) в вероятности классов."""

    def forward(self, X):
        shifted = X - np.max(X, axis=1, keepdims=True)
        exp_values = np.exp(shifted)
        self.P = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.P

    def backward(self, dY):
        """Возвращает градиент по логитам для входящего градиента по вероятностям."""
        projection = np.sum(dY * self.P, axis=1, keepdims=True)
        return self.P * (dY - projection)

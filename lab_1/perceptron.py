import numpy as np


class MLP:
    def __init__(
        self,
        input_size: int,
        hidden_size_1: int = 64,
        hidden_size_2: int = 32,
        output_size: int = 2,
        epochs: int = 100,
        learning_rate: float = 0.01,
        random_state: int = 42,
    ):
        self.learning_rate = learning_rate
        self.epochs = epochs
        rn = np.random.default_rng(random_state)

        self.W1 = rn.standard_normal((input_size, hidden_size_1)) * np.sqrt(
            1 / input_size
        )

        self.b1 = np.zeros((1, hidden_size_1))
        self.W2 = rn.standard_normal((hidden_size_1, hidden_size_2)) * np.sqrt(
            1 / hidden_size_1
        )

        self.b2 = np.zeros((1, hidden_size_2))

        self.W3 = rn.standard_normal((hidden_size_2, output_size)) * np.sqrt(
            1 / hidden_size_2
        )

        self.b3 = np.zeros((1, output_size))

        self.loss_history = []

    def forward(self, X):
        """Прямой проход с сохранением значений для backward."""
        self.X = X

        self.Z1 = X @ self.W1 + self.b1
        self.A1 = self.activation(self.Z1)

        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = self.activation(self.Z2)

        self.Z3 = self.A2 @ self.W3 + self.b3
        self.P = self.softmax(self.Z3)

        return self.P

    def activation(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def activation_derivative(self, a):
        """Производная сигмоиды через уже вычисленную активацию."""
        return a * (1.0 - a)

    def softmax(self, z):
        """Преобразует логиты каждого объекта в вероятности классов."""
        shifted_z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(shifted_z)

        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def loss(self, probabilities, y):
        """Средняя кросс-энтропия."""
        y = np.asarray(y, dtype=int).reshape(-1)
        sample_count = y.shape[0]

        # Не допускаем log(0)
        probabilities = np.clip(probabilities, 1e-12, 1.0)

        # Вероятность правильного класса для каждого объекта
        correct_probabilities = probabilities[
            np.arange(sample_count),
            y,
        ]

        return -np.mean(np.log(correct_probabilities))

    def backward(self, y):
        """Вычисляет градиенты всех весов и bias."""
        y = np.asarray(y, dtype=int).reshape(-1)
        sample_count = y.shape[0]

        # Переводим метки 0/1 в one-hot представление
        y_one_hot = np.zeros_like(self.P)
        y_one_hot[np.arange(sample_count), y] = 1

        # Выходной слой: softmax + cross-entropy
        dZ3 = (self.P - y_one_hot) / sample_count

        self.dW3 = self.A2.T @ dZ3
        self.db3 = np.sum(dZ3, axis=0, keepdims=True)

        # Передаём градиент ко второму скрытому слою
        dA2 = dZ3 @ self.W3.T
        dZ2 = dA2 * self.activation_derivative(self.A2)

        self.dW2 = self.A1.T @ dZ2
        self.db2 = np.sum(dZ2, axis=0, keepdims=True)

        # Передаём градиент к первому скрытому слою
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * self.activation_derivative(self.A1)

        self.dW1 = self.X.T @ dZ1
        self.db1 = np.sum(dZ1, axis=0, keepdims=True)

    def update_parameters(self):
        """Один шаг градиентного спуска."""
        self.W1 -= self.learning_rate * self.dW1
        self.b1 -= self.learning_rate * self.db1

        self.W2 -= self.learning_rate * self.dW2
        self.b2 -= self.learning_rate * self.db2

        self.W3 -= self.learning_rate * self.dW3
        self.b3 -= self.learning_rate * self.db3

    def fit(self, X, y, verbose=False):
        """Обучает сеть методом градиентного спуска."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int).reshape(-1)

        self.loss_history = []
        print_interval = max(1, self.epochs // 10)

        for epoch in range(self.epochs):
            # 1. Прямой проход
            probabilities = self.forward(X)

            # 2. Расчёт ошибки
            current_loss = self.loss(probabilities, y)
            self.loss_history.append(current_loss)

            # 3. Расчёт градиентов
            self.backward(y)

            # 4. Обновление параметров
            self.update_parameters()

            if verbose and (epoch % print_interval == 0 or epoch == self.epochs - 1):
                print(f"Epoch {epoch + 1}/{self.epochs}, loss={current_loss:.6f}")

        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        return self.forward(X)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)

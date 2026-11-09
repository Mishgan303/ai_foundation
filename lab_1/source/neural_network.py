import numpy as np


class NeuralNetwork:
    def __init__(
        self,
        *layers,
        epochs: int = 100,
        learning_rate: float = 0.01,
    ):
        self.layers = list(layers)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.loss_history = []

    def forward(self, X):
        """Выполняет все слои; результат — вероятности."""
        for layer in self.layers:
            X = layer.forward(X)
        self.P = X
        return self.P

    def __call__(self, X):
        """Позволяет выполнить прямой проход через model(X)."""
        return self.forward(X)

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
        """Вычисляет градиенты параметров и возвращает градиент по входу."""
        y = np.asarray(y, dtype=int).reshape(-1)
        sample_count = y.shape[0]

        y_one_hot = np.zeros_like(self.P)
        y_one_hot[np.arange(sample_count), y] = 1

        gradient = (self.P - y_one_hot) / sample_count
        # Производная Softmax уже учтена в формуле выше.
        for layer in reversed(self.layers[:-1]):
            gradient = layer.backward(gradient)
        return gradient

    def update_parameters(self):
        """Обновляет параметры только у обучаемых слоёв после backward."""
        for layer in self.layers:
            if hasattr(layer, "update_parameters"):
                layer.update_parameters(self.learning_rate)

    def fit(self, X, y, verbose=False):
        """Обучает сеть методом градиентного спуска."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int).reshape(-1)

        self.loss_history = []
        print_interval = max(1, self.epochs // 10)

        for epoch in range(self.epochs):
            # 1. Прямой проход
            probabilities = self(X)

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
        return self(X)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)

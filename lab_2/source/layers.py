import numpy as np


class Conv2d:
    """Свёртка: (B, C_in, H, W) → (B, C_out, H_out, W_out)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        rng: np.random.Generator,
        stride: int = 1,
        padding: int = 0,
    ):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Масштаб инициализации для следующего слоя ReLU.
        fan_in = in_channels * kernel_size * kernel_size
        self.K = rng.standard_normal(
            (out_channels, in_channels, kernel_size, kernel_size)
        ) * np.sqrt(2 / fan_in)
        self.b = np.zeros(out_channels)

    def forward(self, X):
        """Собирает окна входа в матрицу и применяет все фильтры сразу."""
        self.input_shape = X.shape
        batch, channels, height, width = X.shape
        p = self.padding
        k = self.kernel_size
        X_padded = np.pad(X, ((0, 0), (0, 0), (p, p), (p, p)))

        out_height = (height + 2 * p - k) // self.stride + 1
        out_width = (width + 2 * p - k) // self.stride + 1
        filters = self.K.shape[0]

        # (B, C, H_out, W_out, k, k): сначала получаем окна, затем учитываем шаг.
        windows = np.lib.stride_tricks.sliding_window_view(
            X_padded, (k, k), axis=(2, 3)
        )[:, :, :: self.stride, :: self.stride]
        # Одна строка — одно окно: (B * H_out * W_out, C * k * k).
        self.columns = windows.transpose(0, 2, 3, 1, 4, 5).reshape(
            batch * out_height * out_width, channels * k * k
        )
        kernels = self.K.reshape(filters, -1)

        # Z = columns @ kernels.T + b; i — элементы окна, f — фильтры.
        output = np.einsum("ni,fi->nf", self.columns, kernels) + self.b
        return output.reshape(batch, out_height, out_width, filters).transpose(
            0, 3, 1, 2
        )

    def backward(self, dY):
        """Сохраняет dK, db и возвращает dX в исходной форме входа."""
        batch, filters, out_height, out_width = dY.shape
        _, channels, height, width = self.input_shape
        k = self.kernel_size
        p = self.padding
        # Та же последовательность окон, что и в self.columns.
        gradient = dY.transpose(0, 2, 3, 1).reshape(-1, filters)

        # dK = gradient.T @ columns; dColumns = gradient @ kernels.
        self.dK = np.einsum(
            "nf,ni->fi", gradient, self.columns, dtype=self.K.dtype
        ).reshape(self.K.shape)
        self.db = np.sum(gradient, axis=0, dtype=self.b.dtype)
        d_columns = np.einsum("nf,fi->ni", gradient, self.K.reshape(filters, -1))
        d_windows = d_columns.reshape(
            batch, out_height, out_width, channels, k, k
        ).transpose(0, 3, 4, 5, 1, 2)

        dX_padded = np.zeros((batch, channels, height + 2 * p, width + 2 * p))
        # col2im: складываем перекрытия; циклы только по k * k смещениям ядра.
        for row in range(k):
            h = slice(row, row + self.stride * out_height, self.stride)
            for column in range(k):
                w = slice(column, column + self.stride * out_width, self.stride)
                dX_padded[:, :, h, w] += d_windows[:, :, row, column]

        # Срез работает и при padding=0: избегаем пустого диапазона 0:-0.
        return dX_padded[:, :, p : p + height, p : p + width]

    def update_parameters(self, learning_rate):
        """Обновляет ядра и смещения после полного обратного прохода."""
        self.K -= learning_rate * self.dK
        self.b -= learning_rate * self.db


class ReLU:
    """Поэлементная активация max(0, X), сохраняющая форму массива."""

    def forward(self, X):
        self.positive_mask = X > 0
        return np.maximum(X, 0)

    def backward(self, dY):
        """В нуле выбираем производную 0; маска пропускает положительные входы."""
        return dY * self.positive_mask


class MaxPool2d:
    """Выбирает максимум окна отдельно для каждого изображения и канала."""

    def __init__(self, kernel_size: int, stride: int):
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, X):
        """Сохраняет форму входа и первую позицию максимума каждого окна."""
        self.input_shape = X.shape
        k = self.kernel_size
        windows = np.lib.stride_tricks.sliding_window_view(
            X, (k, k), axis=(2, 3)
        )[:, :, :: self.stride, :: self.stride]
        # (B, C, H_out, W_out, k * k): каждый квадрат превращаем в вектор.
        windows = windows.reshape(windows.shape[:4] + (k * k,))
        self.max_indices = np.argmax(windows, axis=-1)
        return np.take_along_axis(
            windows, self.max_indices[..., None], axis=-1
        )[..., 0]

    def backward(self, dY):
        """Направляет градиенты в сохранённые позиции максимумов."""
        batch, channels, out_height, out_width = dY.shape
        dX = np.zeros(self.input_shape, dtype=dY.dtype)
        n = np.arange(batch)[:, None, None, None]
        c = np.arange(channels)[None, :, None, None]

        # Индекс внутри окна переводим в координаты исходной карты.
        h = (
            np.arange(out_height)[None, None, :, None] * self.stride
            + self.max_indices // self.kernel_size
        )
        w = (
            np.arange(out_width)[None, None, None, :] * self.stride
            + self.max_indices % self.kernel_size
        )
        # Один пиксель может быть максимумом нескольких перекрывающихся окон.
        np.add.at(dX, (n, c, h, w), dY)
        return dX


class Flatten:
    """Объединяет оси признаков, сохраняя отдельную строку для каждого объекта."""

    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, dY):
        """Возвращает градиенту исходную форму входа без изменения значений."""
        return dY.reshape(self.input_shape)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

from perceptron import MLP

mushroom = fetch_ucirepo(id=73)

X_df = mushroom.data.features.copy()
y_df = mushroom.data.targets.copy()

# Пропуски обозначены знаком "?"
X_df = X_df.replace("?", np.nan).fillna("missing")


X_encoded = pd.get_dummies(
    X_df,
    dtype=np.float64,
)

# e — съедобный: 0
# p — ядовитый: 1
y = y_df.squeeze().map({"e": 0, "p": 1}).to_numpy(dtype=np.int64)

X = X_encoded.to_numpy(dtype=np.float64)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("Весь датасет:", X.shape, y.shape)
print("Обучение:", X_train.shape, y_train.shape)
print("Тест:", X_test.shape, y_test.shape)
print("Количество классов:", np.bincount(y))


model = MLP(
    input_size=X_train.shape[1],
    hidden_size_1=32,
    hidden_size_2=16,
    output_size=2,
    learning_rate=0.2,
    epochs=1750,
    random_state=42,
)

model.fit(X_train, y_train, verbose=True)

y_pred = model.predict(X_test)
# Вероятность положительного класса: 1 — ядовитый гриб.
y_prob = model.predict_proba(X_test)[:, 1]

accuracy = np.mean(y_pred == y_test)
precision = precision_score(y_test, y_pred, pos_label=1)
recall = recall_score(y_test, y_pred, pos_label=1)
f1 = f1_score(y_test, y_pred, pos_label=1)
fpr, tpr, thresholds = roc_curve(y_test, y_prob, pos_label=1)
roc_auc = roc_auc_score(y_test, y_prob)

print(f"Первый loss: {model.loss_history[0]:.6f}")
print(f"Последний loss: {model.loss_history[-1]:.6f}")
print(f"Test accuracy: {accuracy:.4f}")
print(f"Test precision: {precision:.6f}")
print(f"Test recall: {recall:.6f}")
print(f"Test F1: {f1:.6f}")
print(f"Test ROC AUC: {roc_auc:.6f}")


plt.figure()
plt.plot(model.loss_history)
plt.xlabel("Эпоха")
plt.ylabel("Cross-entropy loss")
plt.title("Обучение MLP")
plt.grid()

plt.figure()
plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.6f}")
plt.plot([0, 1], [0, 1], "--", color="gray", label="Случайное ранжирование")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR / Recall)")
plt.title("ROC-кривая MLP: ядовитый класс")
plt.legend()
plt.grid()
plt.show()

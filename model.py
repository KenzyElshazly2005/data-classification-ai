# =========================
# 🌸 Iris ML Project - Full Pipeline
# =========================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# =========================
# 1. Load Dataset
# =========================
iris = load_iris()

X = iris.data
y = iris.target

df = pd.DataFrame(X, columns=iris.feature_names)
df["target"] = y

print("\n📊 Dataset Loaded Successfully!")
print(df.head())

# =========================
# 2. Understand Data (EDA)
# =========================
print("\n📌 Dataset Info:")
print(df.info())

print("\n📌 Description:")
print(df.describe())

print("\n📌 Class names:")
print(iris.target_names)

# =========================
# 3. Data Visualization
# =========================

# 3.1 Pairplot (MOST IMPORTANT VISUAL)
sns.pairplot(df, hue="target")
plt.suptitle("Iris Pairplot", y=1.02)
plt.show()

# 3.2 Class distribution
sns.countplot(x="target", data=df)
plt.title("Class Distribution")
plt.show()

# 3.3 Correlation heatmap
plt.figure(figsize=(8,5))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation")
plt.show()

# =========================
# 4. Data Preprocessing
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# 5. Train/Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# =========================
# 6. Models
# =========================
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=200)
}

# =========================
# 7. Training + Evaluation
# =========================
for name, model in models.items():
    print(f"\n========================")
    print(f"🤖 Model: {name}")
    print(f"========================")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.2f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure()
    sns.heatmap(cm, annot=True, cmap="Blues")
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# =========================
# 8. Final Insight
# =========================
print("\n🎯 Project Completed Successfully!")
print("✔ Data analyzed")
print("✔ Data visualized")
print("✔ Models trained and compared")
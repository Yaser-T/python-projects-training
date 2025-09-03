# Mini Data Science Project - Iris Classification

# 1. Import libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 2. Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Convert to DataFrame for clarity
df = pd.DataFrame(X, columns=iris.feature_names)
df["target"] = y

print("Sample of the dataset:")
print(df.head())

# 3. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train a model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print(
    "\nClassification Report:\n",
    classification_report(y_test, y_pred, target_names=iris.target_names),
)

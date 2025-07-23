# src/train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import pickle

# Load data
df = pd.read_csv("data/iris.csv")
X = df.drop("species", axis=1)
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start MLflow experiment
mlflow.set_experiment("Iris_Classifier")
with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", acc)

    # Save model
    with open("src/model.pkl", "wb") as f:
        pickle.dump(model, f)
    mlflow.log_artifact("src/model.pkl")

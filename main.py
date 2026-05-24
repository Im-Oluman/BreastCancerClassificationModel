import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import pickle

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def load_data(path):
    df = pd.read_csv(path)
    return df


def preprocess_data(df):
    # Drop unnecessary columns
    df = df.drop(columns=["id", "Unnamed: 32"])

    # Encode target
    le = LabelEncoder()
    df["diagnosis"] = le.fit_transform(df["diagnosis"])

    return df


def split_data(df):
    X = df.drop("diagnosis", axis=1)
    y = df["diagnosis"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test


def train_models(X_train, y_train):
    # Scale for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    lr = LogisticRegression()
    lr.fit(X_train_scaled, y_train)

    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)

    return lr, rf, scaler


def evaluate_models(models, X_test, y_test, scaler):
    lr, rf = models

    # Logistic Regression prediction
    X_test_scaled = scaler.transform(X_test)
    lr_pred = lr.predict(X_test_scaled)

    rf_pred = rf.predict(X_test)

    print("\n=== LOGISTIC REGRESSION ===")
    print("Accuracy:", accuracy_score(y_test, lr_pred))
    print(classification_report(y_test, lr_pred))

    print("\n=== RANDOM FOREST ===")
    print("Accuracy:", accuracy_score(y_test, rf_pred))
    print(classification_report(y_test, rf_pred))


def cross_validation(models, X, y):
    lr, rf = models

    lr_scores = cross_val_score(lr, X, y, cv=5)
    rf_scores = cross_val_score(rf, X, y, cv=5)

    print("\n=== CROSS VALIDATION ===")
    print("Logistic Regression CV:", lr_scores.mean())
    print("Random Forest CV:", rf_scores.mean())


def save_model(model, scaler):
    joblib.dump(model, "models/random_forest_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")


def main():
    path = "data/data(2).csv"

    df = load_data(path)
    df = preprocess_data(df)

    X_train, X_test, y_train, y_test = split_data(df)

    lr, rf, scaler = train_models(X_train, y_train)

    evaluate_models((lr, rf), X_test, y_test, scaler)

    X = df.drop("diagnosis", axis=1)
    y = df["diagnosis"]

    cross_validation((lr, rf), X, y)

    save_model(rf, scaler)

    print("\nModel training complete!")


if __name__ == "__main__":
    main()
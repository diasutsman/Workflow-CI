#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by: Dias Utsman
Heart Failure Prediction Model with MLflow Tracking
"""

import os
import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn

# Set random seed for reproducibility
np.random.seed(42)

# Set MLflow tracking URI - using local directory for storage
mlflow.set_tracking_uri(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "mlruns"))


def load_preprocessed_data(data_path):
    """
    Load preprocessed heart failure data from CSV file
    """
    print(f"Loading preprocessed data from: {data_path}")
    data = pd.read_csv(data_path)

    # Split features and target
    X = data.drop('DEATH_EVENT', axis=1)  # DEATH_EVENT is the target column
    y = data['DEATH_EVENT']

    print(f"Data shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")

    return X, y


def split_data(X, y, test_size=0.2):
    """
    Split data into training and testing sets
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    print(f"Training set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """
    Train a Random Forest classifier for heart failure prediction
    """
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight='balanced',  # Since heart failure data might be imbalanced
        random_state=42
    )

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate heart failure prediction model performance
    """
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Return metrics and predictions
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'y_pred': y_pred
    }


def plot_confusion_matrix(cm, classes=['No Death', 'Death']):
    """
    Plot confusion matrix for heart failure prediction
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix - Heart Failure Prediction')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # Save the plot
    os.makedirs('artifacts', exist_ok=True)
    plt.savefig('artifacts/confusion_matrix.png')
    plt.close()


def log_feature_importance(model, feature_names):
    """
    Log feature importance
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(12, 8))
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [
               feature_names[i] for i in indices], rotation=90)
    plt.title('Feature Importances')
    plt.tight_layout()

    # Save the plot
    os.makedirs('artifacts', exist_ok=True)
    plt.savefig('artifacts/feature_importance.png')
    plt.close()

    # Create a DataFrame for easier logging
    importance_df = pd.DataFrame({
        'Feature': [feature_names[i] for i in indices],
        'Importance': importances[indices]
    })

    return importance_df


def main(data_path=None, run_name=None):
    """
    Main function to run the heart failure prediction model training pipeline
    """
    # Define experiment name
    experiment_name = "Heart-Failure-Prediction"
    mlflow.set_experiment(experiment_name)

    # Use provided data_path or default
    if not data_path:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(
            current_dir, "heart_failure_preprocessing", "heart_failure_preprocessed.csv")

    # Start MLflow run with provided run_name or default
    run_name = run_name if run_name else "RandomForest-HeartFailure"
    with mlflow.start_run(run_name=run_name):
        # Load data
        X, y = load_preprocessed_data(data_path)

        # Split data
        X_train, X_test, y_train, y_test = split_data(X, y)

        # Enable MLflow autologging
        mlflow.sklearn.autolog()

        # Train model
        print("\nTraining Random Forest model...")
        model = train_model(X_train, y_train)

        # Evaluate model
        print("\nEvaluating model performance...")
        metrics = evaluate_model(model, X_test, y_test)

        # Plot confusion matrix
        plot_confusion_matrix(metrics['confusion_matrix'])

        # Log feature importance
        importance_df = log_feature_importance(model, X.columns)

        # Log metrics manually (in addition to autolog)
        mlflow.log_metric("accuracy", metrics['accuracy'])
        mlflow.log_metric("precision", metrics['precision'])
        mlflow.log_metric("recall", metrics['recall'])
        mlflow.log_metric("f1", metrics['f1'])

        # Log artifacts
        mlflow.log_artifact("artifacts/confusion_matrix.png")
        mlflow.log_artifact("artifacts/feature_importance.png")

        # Log feature importance as CSV
        importance_path = "artifacts/feature_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        mlflow.log_artifact(importance_path)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        print(f"\nModel training completed successfully!")
        print(f"Model artifacts logged to MLflow")
        print(f"Run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train a heart failure prediction model with MLflow tracking")
    parser.add_argument("--data_path", type=str,
                        help="Path to the preprocessed heart failure dataset")
    parser.add_argument("--run_name", type=str, help="Name for the MLflow run")
    args = parser.parse_args()

    # Run the main function with parsed arguments
    main(data_path=args.data_path, run_name=args.run_name)

#!/usr/bin/env python3
"""
Modelling Script - Dewangga Megananda
Proyek Akhir Machine Learning Dicoding - Skilled Level

Script untuk training model dengan MLflow tracking.
Model: RandomForestClassifier dengan logging parameter dan metric.
"""

import pandas as pd
import numpy as np
import os
import sys
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from mlflow.tracking import MlflowClient

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('modelling.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class MLflowModelTrainer:
    """Class untuk training model dengan MLflow tracking"""

    def __init__(self, experiment_name="Dewangga_Megananda_Experiment"):
        self.experiment_name = experiment_name
        self.client = MlflowClient()

        # Set experiment
        try:
            mlflow.set_experiment(experiment_name)
            logger.info(f"Set MLflow experiment: {experiment_name}")
        except Exception as e:
            logger.warning(f"Could not set experiment: {e}")

    def load_data(self, data_dir):
        """Load training dan testing data dari separate files"""
        try:
            # Load training data
            X_train = pd.read_csv(f"{data_dir}/X_train.csv")
            y_train = pd.read_csv(f"{data_dir}/y_train.csv").iloc[:, 0]  # Get first column
            train_data = pd.concat([X_train, y_train], axis=1)
            train_data.columns = list(X_train.columns) + ['target']

            # Load test data
            X_test = pd.read_csv(f"{data_dir}/X_test.csv")
            y_test = pd.read_csv(f"{data_dir}/y_test.csv").iloc[:, 0]  # Get first column
            test_data = pd.concat([X_test, y_test], axis=1)
            test_data.columns = list(X_test.columns) + ['target']

            logger.info(f"Training data shape: {train_data.shape}")
            logger.info(f"Test data shape: {test_data.shape}")

            return train_data, test_data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def prepare_features_target(self, data, target_column='target'):
        """Pisahkan features dan target"""
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")

        X = data.drop(columns=[target_column])
        y = data[target_column]

        logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
        return X, y

    def train_model(self, X_train, y_train, model_params=None):
        """Train RandomForestClassifier model"""
        if model_params is None:
            model_params = {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42,
                'n_jobs': -1
            }

        logger.info(f"Training model with parameters: {model_params}")

        model = RandomForestClassifier(**model_params)
        model.fit(X_train, y_train)

        logger.info("Model training completed")
        return model

    def evaluate_model(self, model, X_test, y_test):
        """Evaluasi model dan hitung metrics"""
        logger.info("Evaluating model...")

        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision_macro': precision_score(y_test, y_pred, average='macro'),
            'recall_macro': recall_score(y_test, y_pred, average='macro'),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'precision_weighted': precision_score(y_test, y_pred, average='weighted'),
            'recall_weighted': recall_score(y_test, y_pred, average='weighted'),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted')
        }

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        logger.info(f"Model Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"F1 Score (Macro): {metrics['f1_macro']:.4f}")

        return metrics, cm, y_pred, y_pred_proba

    def log_to_mlflow(self, model, model_params, metrics, cm, X_train, y_train, run_name="RandomForest_Baseline"):
        """Log semua informasi ke MLflow"""
        with mlflow.start_run(run_name=run_name) as run:
            # Log parameters
            mlflow.log_params(model_params)

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log model info (parameters sudah di-log di atas)
            mlflow.log_param("model_type", "RandomForestClassifier")
            mlflow.log_param("n_features", X_train.shape[1])
            mlflow.log_param("n_classes", len(np.unique(y_train)))

            # Create dan log confusion matrix plot
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['setosa', 'versicolor', 'virginica'],
                       yticklabels=['setosa', 'versicolor', 'virginica'])
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')

            # Save plot temporarily dan log sebagai artifact
            cm_path = "confusion_matrix.png"
            plt.savefig(cm_path)
            mlflow.log_artifact(cm_path, "plots")
            plt.close()

            # Log feature importance jika ada
            if hasattr(model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': [f'feature_{i}' for i in range(len(model.feature_importances_))],
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)

                # Save feature importance
                fi_path = "feature_importance.csv"
                feature_importance.to_csv(fi_path, index=False)
                mlflow.log_artifact(fi_path, "feature_importance")

            # Save model using MLflow
            mlflow.sklearn.log_model(model, "model")

            run_id = run.info.run_id
            logger.info(f"Logged run to MLflow with ID: {run_id}")

            return run_id

    def cross_validate_model(self, model, X_train, y_train, cv=5):
        """Cross validation untuk evaluasi robust"""
        logger.info(f"Performing {cv}-fold cross validation...")

        cv_accuracy = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
        cv_precision = cross_val_score(model, X_train, y_train, cv=cv, scoring='precision_macro')
        cv_recall = cross_val_score(model, X_train, y_train, cv=cv, scoring='recall_macro')
        cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_macro')

        cv_metrics = {
            'cv_accuracy_mean': cv_accuracy.mean(),
            'cv_accuracy_std': cv_accuracy.std(),
            'cv_precision_mean': cv_precision.mean(),
            'cv_precision_std': cv_precision.std(),
            'cv_recall_mean': cv_recall.mean(),
            'cv_recall_std': cv_recall.std(),
            'cv_f1_mean': cv_f1.mean(),
            'cv_f1_std': cv_f1.std()
        }

        logger.info(f"CV Accuracy: {cv_metrics['cv_accuracy_mean']:.4f} (+/- {cv_metrics['cv_accuracy_std']:.4f})")
        logger.info(f"CV F1 Score: {cv_metrics['cv_f1_mean']:.4f} (+/- {cv_metrics['cv_f1_std']:.4f})")

        return cv_metrics


def main():
    """Main function untuk training model"""
    # Inisialisasi trainer
    trainer = MLflowModelTrainer()

    try:
        # Load data
        train_data, test_data = trainer.load_data(
            '../Eksperimen_SML_Dewangga_Megananda/preprocessing/dataset_preprocessing'
        )

        # Prepare features dan target
        X_train, y_train = trainer.prepare_features_target(train_data)
        X_test, y_test = trainer.prepare_features_target(test_data)

        # Model parameters
        model_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42,
            'n_jobs': -1
        }

        # Train model
        model = trainer.train_model(X_train, y_train, model_params)

        # Cross validation
        cv_metrics = trainer.cross_validate_model(model, X_train, y_train)

        # Evaluate model
        metrics, cm, y_pred, y_pred_proba = trainer.evaluate_model(model, X_test, y_test)

        # Combine all metrics
        all_metrics = {**metrics, **cv_metrics}

        # Log to MLflow
        run_id = trainer.log_to_mlflow(model, model_params, all_metrics, cm, X_train, y_train)

        logger.info("Model training and logging completed successfully!")
        logger.info(f"MLflow Run ID: {run_id}")

        # Print classification report
        print("\\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['setosa', 'versicolor', 'virginica']))

    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Hyperparameter Tuning Script - Dewangga Megananda
Proyek Akhir Machine Learning Dicoding - Skilled Level

Script untuk hyperparameter tuning dengan MLflow tracking.
Minimal 3 kombinasi parameter yang berbeda.
"""

import pandas as pd
import numpy as np
import os
import sys
import logging
import itertools
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from mlflow.tracking import MlflowClient

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('modelling_tuning.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """Class untuk hyperparameter tuning dengan MLflow tracking"""

    def __init__(self, experiment_name="Dewangga_Megananda_Hyperparameter_Tuning"):
        self.experiment_name = experiment_name
        self.client = MlflowClient()

        # Set experiment
        try:
            mlflow.set_experiment(experiment_name)
            logger.info(f"Set MLflow experiment: {experiment_name}")
        except Exception as e:
            logger.warning(f"Could not set experiment: {e}")

    def get_parameter_combinations(self):
        """Generate minimal 3 kombinasi parameter yang berbeda"""

        # Parameter grid untuk RandomForest
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }

        # Generate semua kombinasi
        all_combinations = list(itertools.product(
            param_grid['n_estimators'],
            param_grid['max_depth'],
            param_grid['min_samples_split'],
            param_grid['min_samples_leaf']
        ))

        # Pilih 5 kombinasi yang beragam untuk demo
        selected_combinations = [
            # Kombinasi 1: Default-like parameters
            {'n_estimators': 100, 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1},
            # Kombinasi 2: Conservative parameters
            {'n_estimators': 50, 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 2},
            # Kombinasi 3: Aggressive parameters
            {'n_estimators': 200, 'max_depth': 20, 'min_samples_split': 2, 'min_samples_leaf': 1},
            # Kombinasi 4: Balanced parameters
            {'n_estimators': 100, 'max_depth': 15, 'min_samples_split': 3, 'min_samples_leaf': 1},
            # Kombinasi 5: Shallow tree parameters
            {'n_estimators': 150, 'max_depth': 5, 'min_samples_split': 2, 'min_samples_leaf': 2}
        ]

        logger.info(f"Generated {len(selected_combinations)} parameter combinations")
        return selected_combinations

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

    def train_model(self, X_train, y_train, model_params):
        """Train RandomForestClassifier dengan parameter tertentu"""
        logger.info(f"Training model with parameters: {model_params}")

        model = RandomForestClassifier(**model_params, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        logger.info("Model training completed")
        return model

    def evaluate_model(self, model, X_test, y_test):
        """Evaluasi model dan hitung metrics"""
        logger.info("Evaluating model...")

        # Predictions
        y_pred = model.predict(X_test)

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

        logger.info(".4f")
        logger.info(".4f")

        return metrics, cm, y_pred

    def log_to_mlflow(self, model, model_params, metrics, cm, run_name):
        """Log semua informasi ke MLflow"""
        with mlflow.start_run(run_name=run_name) as run:
            # Log parameters
            mlflow.log_params(model_params)

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log model info
            mlflow.log_param("model_type", "RandomForestClassifier")

            # Create dan log confusion matrix plot
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['setosa', 'versicolor', 'virginica'],
                       yticklabels=['setosa', 'versicolor', 'virginica'])
            plt.title(f'Confusion Matrix - {run_name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')

            # Save plot temporarily dan log sebagai artifact
            cm_path = f"confusion_matrix_{run_name.replace(' ', '_')}.png"
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
                fi_path = f"feature_importance_{run_name.replace(' ', '_')}.csv"
                feature_importance.to_csv(fi_path, index=False)
                mlflow.log_artifact(fi_path, "feature_importance")

            run_id = run.info.run_id
            logger.info(f"Logged run '{run_name}' to MLflow with ID: {run_id}")

            return run_id

    def perform_hyperparameter_tuning(self, X_train, y_train, X_test, y_test):
        """Lakukan hyperparameter tuning untuk semua kombinasi parameter"""
        logger.info("Starting hyperparameter tuning...")

        # Get parameter combinations
        param_combinations = self.get_parameter_combinations()

        tuning_results = []

        for i, params in enumerate(param_combinations, 1):
            logger.info(f"\\n--- Tuning Run {i}/{len(param_combinations)} ---")
            logger.info(f"Parameters: {params}")

            # Train model
            model = self.train_model(X_train, y_train, params)

            # Evaluate model
            metrics, cm, y_pred = self.evaluate_model(model, X_test, y_test)

            # Create run name
            run_name = f"Tuning_Run_{i}_n{params['n_estimators']}_d{params['max_depth']}_s{params['min_samples_split']}_l{params['min_samples_leaf']}"

            # Log to MLflow
            run_id = self.log_to_mlflow(model, params, metrics, cm, run_name)

            # Store results
            result = {
                'run_id': run_id,
                'run_name': run_name,
                'parameters': params,
                'metrics': metrics
            }
            tuning_results.append(result)

            logger.info(f"Completed run {i} with F1 Score: {metrics['f1_macro']:.4f}")

        return tuning_results

    def get_best_model(self, tuning_results):
        """Get best model berdasarkan F1 score"""
        best_result = max(tuning_results, key=lambda x: x['metrics']['f1_macro'])

        logger.info("\\n=== BEST MODEL ===")
        logger.info(f"Run Name: {best_result['run_name']}")
        logger.info(f"F1 Score: {best_result['metrics']['f1_macro']:.4f}")
        logger.info(f"Parameters: {best_result['parameters']}")

        return best_result

    def save_best_model(self, best_result, X_train, y_train):
        """Save best model menggunakan MLflow Model Registry"""
        try:
            logger.info("Saving best model to MLflow Model Registry...")

            # Retrain best model dengan full training data
            best_params = best_result['parameters']
            best_model = self.train_model(X_train, y_train, best_params)

            # Register model ke MLflow Model Registry
            with mlflow.start_run(run_name="Best_Model_Registration") as run:
                # Log parameters dan metrics
                mlflow.log_params(best_params)
                mlflow.log_metrics(best_result['metrics'])

                # Log model dengan specific name untuk inference
                mlflow.sklearn.log_model(
                    sk_model=best_model,
                    artifact_path="model",
                    registered_model_name="Dewangga_RF_Model"
                )

                logger.info("Best model saved to MLflow Model Registry as 'Dewangga_RF_Model'")
                logger.info("Model can be loaded for inference using: models:/Dewangga_RF_Model/Production")

        except Exception as e:
            logger.error(f"Error saving best model: {e}")
            raise


def main():
    """Main function untuk hyperparameter tuning"""
    # Inisialisasi tuner
    tuner = HyperparameterTuner()

    try:
        # Load data
        train_data, test_data = tuner.load_data(
            '../Eksperimen_SML_Dewangga_Megananda/preprocessing/dataset_preprocessing'
        )

        # Prepare features dan target
        X_train, y_train = tuner.prepare_features_target(train_data)
        X_test, y_test = tuner.prepare_features_target(test_data)

        # Perform hyperparameter tuning
        tuning_results = tuner.perform_hyperparameter_tuning(X_train, y_train, X_test, y_test)

        # Get best model
        best_model = tuner.get_best_model(tuning_results)

        # Save best model to MLflow Model Registry
        tuner.save_best_model(best_model, X_train, y_train)

        logger.info("\\nHyperparameter tuning completed successfully!")
        logger.info(f"Total runs: {len(tuning_results)}")
        logger.info(f"Best model: {best_model['run_name']}")
        logger.info("Best model saved to MLflow Model Registry")

        # Print summary
        print("\\n=== TUNING SUMMARY ===")
        print(f"{'Run':<30} {'F1 Score':<10} {'Accuracy':<10}")
        print("-" * 50)
        for result in tuning_results:
            print(f"{result['run_name']:<30} {result['metrics']['f1_macro']:<10.4f} {result['metrics']['accuracy']:<10.4f}")

    except Exception as e:
        logger.error(f"Tuning failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
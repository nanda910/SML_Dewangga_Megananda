#!/usr/bin/env python3
"""
Automate Preprocessing Script - Dewangga Megananda
Proyek Akhir Machine Learning Dicoding - Skilled Level

Script otomatisasi untuk preprocessing data yang dapat digunakan
untuk berbagai dataset machine learning.

Usage:
    python automate_Dewangga_Megananda.py --input path/to/dataset.csv --output path/to/output/
"""

import pandas as pd
import numpy as np
import argparse
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocessing.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Class untuk preprocessing data otomatis"""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.label_encoders = {}

    def load_data(self, file_path):
        """Load data dari file CSV"""
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                raise ValueError("Unsupported file format. Please use CSV files.")

            logger.info(f"Data loaded successfully from {file_path}")
            logger.info(f"Dataset shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def handle_missing_values(self, df):
        """Handle missing values dalam dataset"""
        logger.info("Handling missing values...")

        # Cek missing values
        missing_before = df.isnull().sum().sum()
        logger.info(f"Total missing values before imputation: {missing_before}")

        # Identifikasi kolom numerik dan kategorikal
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns

        # Impute kolom numerik dengan mean
        if len(numeric_columns) > 0:
            df[numeric_columns] = self.imputer.fit_transform(df[numeric_columns])
            logger.info(f"Imputed {len(numeric_columns)} numeric columns with mean strategy")

        # Impute kolom kategorikal dengan mode
        for col in categorical_columns:
            if df[col].isnull().sum() > 0:
                mode_value = df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown'
                df[col].fillna(mode_value, inplace=True)
                logger.info(f"Imputed categorical column '{col}' with mode: {mode_value}")

        missing_after = df.isnull().sum().sum()
        logger.info(f"Total missing values after imputation: {missing_after}")

        return df

    def encode_categorical_features(self, df):
        """Encode fitur kategorikal menjadi numerik"""
        logger.info("Encoding categorical features...")

        categorical_columns = df.select_dtypes(include=['object', 'category']).columns

        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()

            df[col] = self.label_encoders[col].fit_transform(df[col])
            logger.info(f"Encoded categorical column '{col}' with {len(self.label_encoders[col].classes_)} classes")

        return df

    def scale_features(self, df, exclude_columns=None):
        """Scale fitur numerik menggunakan StandardScaler"""
        if exclude_columns is None:
            exclude_columns = []

        logger.info("Scaling numeric features...")

        # Identifikasi kolom yang akan di-scale
        numeric_columns = [col for col in df.select_dtypes(include=[np.number]).columns
                          if col not in exclude_columns]

        if len(numeric_columns) > 0:
            df_scaled = df.copy()
            df_scaled[numeric_columns] = self.scaler.fit_transform(df[numeric_columns])
            logger.info(f"Scaled {len(numeric_columns)} numeric columns")
            return df_scaled
        else:
            logger.warning("No numeric columns found to scale")
            return df

    def split_data(self, X, y, test_size=0.2, stratify=None):
        """Split data menjadi training dan testing set"""
        logger.info(f"Splitting data with test_size={test_size}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state,
            stratify=stratify
        )

        logger.info(f"Training set shape: {X_train.shape}")
        logger.info(f"Test set shape: {X_test.shape}")

        return X_train, X_test, y_train, y_test

    def save_data(self, data_dict, output_dir):
        """Save data ke file CSV"""
        logger.info(f"Saving processed data to {output_dir}")

        # Buat direktori output jika belum ada
        os.makedirs(output_dir, exist_ok=True)

        for filename, data in data_dict.items():
            output_path = os.path.join(output_dir, filename)
            if isinstance(data, pd.DataFrame):
                data.to_csv(output_path, index=False)
            elif isinstance(data, pd.Series):
                data.to_csv(output_path, index=False)
            else:
                # Untuk numpy arrays
                pd.DataFrame(data).to_csv(output_path, index=False)
            logger.info(f"Saved {filename} to {output_path}")

    def preprocess_pipeline(self, input_file, output_dir, target_column=None):
        """Pipeline lengkap preprocessing"""
        logger.info("Starting preprocessing pipeline...")

        # 1. Load data
        df = self.load_data(input_file)

        # 2. Handle missing values
        df = self.handle_missing_values(df)

        # 3. Encode categorical features
        df = self.encode_categorical_features(df)

        # 4. Split features and target
        if target_column:
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in dataset")

            X = df.drop(columns=[target_column])
            y = df[target_column]
            logger.info(f"Separated features (shape: {X.shape}) and target (shape: {y.shape})")
        else:
            X = df
            y = None
            logger.info("No target column specified, processing all columns as features")

        # 5. Scale features
        X_scaled = self.scale_features(X)

        # 6. Split data jika ada target
        if y is not None:
            X_train, X_test, y_train, y_test = self.split_data(
                X_scaled, y, stratify=y if y.nunique() < 10 else None
            )

            # 7. Save all data
            data_to_save = {
                'X_train.csv': X_train,
                'X_test.csv': X_test,
                'y_train.csv': y_train,
                'y_test.csv': y_test,
                'X_scaled.csv': X_scaled,
                'y.csv': y
            }
        else:
            data_to_save = {
                'X_scaled.csv': X_scaled
            }

        self.save_data(data_to_save, output_dir)
        logger.info("Preprocessing pipeline completed successfully!")

        return X_scaled, y


def main():
    """Main function untuk menjalankan script"""
    parser = argparse.ArgumentParser(
        description='Automate Data Preprocessing - Dewangga Megananda'
    )
    parser.add_argument('--input', required=True, help='Path to input CSV file')
    parser.add_argument('--output', required=True, help='Path to output directory')
    parser.add_argument('--target', help='Name of target column (optional)')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size (default: 0.2)')

    args = parser.parse_args()

    # Validasi input
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    # Inisialisasi preprocessor
    preprocessor = DataPreprocessor()

    try:
        # Jalankan pipeline
        X_scaled, y = preprocessor.preprocess_pipeline(
            args.input, args.output, args.target
        )

        logger.info("Preprocessing completed successfully!")
        logger.info(f"Scaled features shape: {X_scaled.shape}")
        if y is not None:
            logger.info(f"Target shape: {y.shape}")

    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
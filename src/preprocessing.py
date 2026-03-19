"""Data preprocessing utilities."""

from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from typing import Any

from src.config import FEATURE_NAMES, CATEGORICAL_COLUMNS, TARGET_COLUMN, CHURN_LABELS


class DataPreprocessor:
    """Handles all data preprocessing operations."""

    def __init__(self) -> None:
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.scaler: StandardScaler | None = None
        self.feature_names: list[str] = FEATURE_NAMES

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load data from CSV file."""
        df = pd.read_csv(filepath)
        return df

    def prepare_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict[str, LabelEncoder]]:
        """Prepare data for model training."""
        df_processed = df.drop("id_cliente", axis=1)
        
        df_processed[TARGET_COLUMN] = df_processed[TARGET_COLUMN].map(CHURN_LABELS)
        
        for col in CATEGORICAL_COLUMNS:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col])
            self.label_encoders[col] = le
        
        X = df_processed.drop(TARGET_COLUMN, axis=1)
        y = df_processed[TARGET_COLUMN]
        
        return X, y, self.label_encoders

    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train and test sets."""
        return train_test_split(
            X.values, y.values, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y
        )

    def scale_features(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
        """Scale features using StandardScaler."""
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, self.scaler

    def transform_input(self, data: dict[str, Any]) -> np.ndarray:
        """Transform input data for prediction."""
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call scale_features first.")
        
        data_processed = data.copy()
        for col in CATEGORICAL_COLUMNS:
            if col in data_processed:
                data_processed[col] = self.label_encoders[col].transform([data_processed[col]])[0]
        
        df = pd.DataFrame([data_processed])[self.feature_names]
        return self.scaler.transform(df.values)

    def save_preprocessors(self, path: str) -> None:
        """Save preprocessors to disk."""
        import joblib
        joblib.dump(self.label_encoders, f"{path}/label_encoders.pkl")
        joblib.dump(self.scaler, f"{path}/scaler.pkl")

    def load_preprocessors(self, path: str) -> None:
        """Load preprocessors from disk."""
        import joblib
        self.label_encoders = joblib.load(f"{path}/label_encoders.pkl")
        self.scaler = joblib.load(f"{path}/scaler.pkl")

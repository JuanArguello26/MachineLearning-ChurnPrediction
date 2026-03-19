"""Model training and evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    AccuracyScore,
    ConfusionMatrix,
    F1Score,
    PrecisionScore,
    RecallScore,
    ROC_AUC,
    ClassificationReport,
)
from typing import Any

from src.config import MODEL_CONFIG
from src.preprocessing import DataPreprocessor


class ChurnModel:
    """Churn prediction model wrapper."""

    def __init__(self) -> None:
        self.model: RandomForestClassifier | None = None
        self.preprocessor = DataPreprocessor()

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> None:
        """Train the Random Forest model."""
        self.model = RandomForestClassifier(
            n_estimators=MODEL_CONFIG.n_estimators,
            max_depth=MODEL_CONFIG.max_depth,
            random_state=MODEL_CONFIG.random_state,
            class_weight=MODEL_CONFIG.class_weight,
        )
        self.model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict_proba(X)

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> dict[str, float]:
        """Evaluate model performance."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        y_pred = self.model.predict(X_test)
        y_proba = self.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": AccuracyScore()(y_test, y_pred),
            "precision": PrecisionScore()(y_test, y_pred),
            "recall": RecallScore()(y_test, y_pred),
            "f1_score": F1Score()(y_test, y_pred),
            "roc_auc": ROC_AUC()(y_test, y_proba),
        }
        return metrics

    def get_classification_report(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> str:
        """Get detailed classification report."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        y_pred = self.model.predict(X_test)
        return ClassificationReport()(
            y_test, y_pred, target_names=["No Churn", "Churn"]
        )

    def get_confusion_matrix(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> np.ndarray:
        """Get confusion matrix."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        y_pred = self.model.predict(X_test)
        return ConfusionMatrix()(y_test, y_pred)

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance as DataFrame."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return pd.DataFrame({
            "feature": self.preprocessor.feature_names,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False)

    def save_model(self, path: str) -> None:
        """Save model to disk."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        import joblib
        joblib.dump(self.model, f"{path}/churn_model.pkl")

    def load_model(self, path: str) -> None:
        """Load model from disk."""
        import joblib
        self.model = joblib.load(f"{path}/churn_model.pkl")

    def full_pipeline(
        self,
        data_path: str,
        save_path: str = "models",
    ) -> dict[str, Any]:
        """Run complete training pipeline."""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        df = self.preprocessor.load_data(data_path)
        X, y, _ = self.preprocessor.prepare_data(df)
        
        X_train, X_test, y_train, y_test = self.preprocessor.split_data(X, y)
        X_train_scaled, X_test_scaled, scaler = self.preprocessor.scale_features(X_train, X_test)
        
        self.train(X_train_scaled, y_train)
        
        metrics = self.evaluate(X_test_scaled, y_test)
        
        self.save_model(save_path)
        self.preprocessor.save_preprocessors(save_path)
        
        return metrics

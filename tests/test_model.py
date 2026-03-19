"""Tests for ChurnModel."""

from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import DataPreprocessor
from src.model import ChurnModel


@pytest.fixture
def data_path() -> str:
    """Get path to test data."""
    return "data/churn_data.csv"


@pytest.fixture
def sample_dataframe(data_path: str) -> pd.DataFrame:
    """Load sample data for testing."""
    return pd.read_csv(data_path)


@pytest.fixture
def preprocessor() -> DataPreprocessor:
    """Create preprocessor instance."""
    return DataPreprocessor()


class TestDataLoading:
    """Tests for data loading functionality."""

    def test_data_file_exists(self, data_path: str) -> None:
        """Test that data file exists."""
        assert os.path.exists(data_path), "Data file not found"

    def test_data_not_empty(self, sample_dataframe: pd.DataFrame) -> None:
        """Test that data is not empty."""
        assert len(sample_dataframe) > 0, "Data is empty"

    def test_data_has_required_columns(self, sample_dataframe: pd.DataFrame) -> None:
        """Test that all required columns are present."""
        required: list[str] = [
            "id_cliente", "edad", "genero", "estado_civil", "tipo_plan",
            "meses_contrato", "factura_mensual", "GB_consumidos",
            "llamadas_soporte", "cambios_plan", "churn",
        ]
        for col in required:
            assert col in sample_dataframe.columns, f"Missing column: {col}"


class TestDataPreparation:
    """Tests for data preparation functionality."""

    def test_prepare_data(self, preprocessor: DataPreprocessor, sample_dataframe: pd.DataFrame) -> None:
        """Test data preparation."""
        X, y, encoders = preprocessor.prepare_data(sample_dataframe)
        assert X.shape[0] == len(y), "X and y length mismatch"
        assert len(encoders) == 3, "Should have 3 encoders"

    def test_churn_values(self, sample_dataframe: pd.DataFrame) -> None:
        """Test churn column values."""
        assert set(sample_dataframe["churn"].unique()) == {"Si", "No"}, "Invalid churn values"


class TestModel:
    """Tests for model functionality."""

    def test_model_training(self, preprocessor: DataPreprocessor, sample_dataframe: pd.DataFrame) -> None:
        """Test model training."""
        X, y, _ = preprocessor.prepare_data(sample_dataframe)
        X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
        X_train_scaled, X_test_scaled, _ = preprocessor.scale_features(X_train, X_test)

        model = ChurnModel()
        model.train(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        from sklearn.metrics import accuracy_score
        assert accuracy_score(y_test, y_pred) > 0.5, "Model accuracy too low"

    def test_model_predictions(self, preprocessor: DataPreprocessor, sample_dataframe: pd.DataFrame) -> None:
        """Test model predictions."""
        X, y, _ = preprocessor.prepare_data(sample_dataframe)
        X_train, X_test, y_train, _ = preprocessor.split_data(X, y)
        X_train_scaled, X_test_scaled, _ = preprocessor.scale_features(X_train, X_test)

        model = ChurnModel()
        model.train(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled[:5])
        y_proba = model.predict_proba(X_test_scaled[:5])

        assert len(y_pred) == 5, "Wrong prediction length"
        assert y_proba.shape == (5, 2), "Wrong probability shape"
        assert ((y_proba >= 0).all() and (y_proba <= 1).all()), "Probabilities out of range"

    def test_model_evaluation(self, preprocessor: DataPreprocessor, sample_dataframe: pd.DataFrame) -> None:
        """Test model evaluation."""
        X, y, _ = preprocessor.prepare_data(sample_dataframe)
        X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
        X_train_scaled, X_test_scaled, _ = preprocessor.scale_features(X_train, X_test)

        model = ChurnModel()
        model.train(X_train_scaled, y_train)
        
        metrics = model.evaluate(X_test_scaled, y_test)
        
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "roc_auc" in metrics


class TestAPIPayload:
    """Tests for API payload structure."""

    def test_customer_data_structure(self) -> None:
        """Test customer data structure for API."""
        sample: dict[str, int | float | str] = {
            "edad": 30,
            "genero": "Masculino",
            "estado_civil": "Soltero",
            "tipo_plan": "Premium",
            "meses_contrato": 12,
            "factura_mensual": 50.0,
            "GB_consumidos": 30.0,
            "llamadas_soporte": 2,
            "cambios_plan": 0,
        }

        required_keys: list[str] = list(sample.keys())
        assert len(required_keys) == 9, "Wrong number of fields"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

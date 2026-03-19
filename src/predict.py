"""Prediction utilities for production use."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


from src.preprocessing import DataPreprocessor


@dataclass
class PredictionResult:
    """Container for prediction results."""
    
    churn: bool
    churn_probability: float
    mensaje: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "churn": self.churn,
            "churn_probability": self.churn_probability,
            "mensaje": self.mensaje,
        }


class ChurnPredictor:
    """Handles churn predictions for new customers."""

    def __init__(self, model: Any, preprocessor: DataPreprocessor) -> None:
        self.model = model
        self.preprocessor = preprocessor

    def predict(self, customer_data: dict[str, Any]) -> PredictionResult:
        """Make a prediction for a single customer."""
        X = self.preprocessor.transform_input(customer_data)
        
        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0][1]
        
        mensaje = (
            "El cliente probablemente cancelará"
            if prediction
            else "El cliente probablemente permanecerá"
        )
        
        return PredictionResult(
            churn=bool(prediction),
            churn_probability=round(float(probability), 4),
            mensaje=mensaje,
        )

    def batch_predict(self, customers: list[dict[str, Any]]) -> list[PredictionResult]:
        """Make predictions for multiple customers."""
        return [self.predict(customer) for customer in customers]


def load_predictor(
    model_path: str = "models/churn_model.pkl",
    preprocessors_path: str = "models",
) -> ChurnPredictor:
    """Load a predictor from saved artifacts."""
    import joblib
    
    model = joblib.load(model_path)
    preprocessor = DataPreprocessor()
    preprocessor.load_preprocessors(preprocessors_path)
    
    return ChurnPredictor(model=model, preprocessor=preprocessor)

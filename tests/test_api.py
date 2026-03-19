"""Tests for Churn Prediction API."""

from __future__ import annotations

import os
import sys
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import app


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Create test client."""
    return TestClient(app)


class TestAPI:
    """Tests for API endpoints."""

    def test_root(self, client: TestClient) -> None:
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()

    def test_health_check(self, client: TestClient) -> None:
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


class TestPredictEndpoint:
    """Tests for prediction endpoint."""

    @patch("api.main.model")
    @patch("api.main.preprocessor")
    def test_predict_churn_no(
        self, mock_preprocessor: MagicMock, mock_model: MagicMock, client: TestClient
    ) -> None:
        """Test prediction when customer will not churn."""
        mock_model.predict.return_value = [0]
        mock_model.predict_proba.return_value = [[0.8, 0.2]]
        mock_preprocessor.transform_input.return_value = [[1, 2, 3, 4, 5, 6, 7, 8, 9]]

        payload: dict[str, int | float | str] = {
            "edad": 35,
            "genero": "Masculino",
            "estado_civil": "Casado",
            "tipo_plan": "Premium",
            "meses_contrato": 24,
            "factura_mensual": 80.0,
            "GB_consumidos": 45.0,
            "llamadas_soporte": 1,
            "cambios_plan": 0,
        }

        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "churn" in data
        assert "churn_probability" in data
        assert "mensaje" in data
        assert data["churn"] is False
        assert data["churn_probability"] == 0.2

    @patch("api.main.model")
    @patch("api.main.preprocessor")
    def test_predict_churn_yes(
        self, mock_preprocessor: MagicMock, mock_model: MagicMock, client: TestClient
    ) -> None:
        """Test prediction when customer will churn."""
        mock_model.predict.return_value = [1]
        mock_model.predict_proba.return_value = [[0.3, 0.7]]
        mock_preprocessor.transform_input.return_value = [[1, 2, 3, 4, 5, 6, 7, 8, 9]]

        payload: dict[str, int | float | str] = {
            "edad": 25,
            "genero": "Femenino",
            "estado_civil": "Soltero",
            "tipo_plan": "Basico",
            "meses_contrato": 3,
            "factura_mensual": 40.0,
            "GB_consumidos": 15.0,
            "llamadas_soporte": 5,
            "cambios_plan": 2,
        }

        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["churn"] is True
        assert data["churn_probability"] == 0.7
        assert "cancelará" in data["mensaje"]

    def test_predict_invalid_data(self, client: TestClient) -> None:
        """Test prediction with invalid data."""
        payload: dict[str, str] = {
            "edad": "invalid",
            "genero": "Masculino",
        }

        response = client.post("/predict", json=payload)
        assert response.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

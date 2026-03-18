import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import app

client = TestClient(app)

class TestAPI:
    def test_root(self):
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()
    
    def test_health_check(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

class TestPredictEndpoint:
    @patch('api.main.model')
    @patch('api.main.scaler')
    @patch('api.main.label_encoders')
    def test_predict_churn_no(self, mock_encoders, mock_scaler, mock_model):
        mock_model.predict.return_value = [0]
        mock_model.predict_proba.return_value = [[0.8, 0.2]]
        mock_scaler.transform.return_value = [[1, 2, 3, 4, 5, 6, 7, 8, 9]]
        mock_encoders.__getitem__.return_value.transform.return_value = [0]
        
        payload = {
            "edad": 35,
            "genero": "Masculino",
            "estado_civil": "Casado",
            "tipo_plan": "Premium",
            "meses_contrato": 24,
            "factura_mensual": 80.0,
            "GB_consumidos": 45.0,
            "llamadas_soporte": 1,
            "cambios_plan": 0
        }
        
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "churn" in data
        assert "churn_probability" in data
    
    @patch('api.main.model')
    @patch('api.main.scaler')
    @patch('api.main.label_encoders')
    def test_predict_churn_yes(self, mock_encoders, mock_scaler, mock_model):
        mock_model.predict.return_value = [1]
        mock_model.predict_proba.return_value = [[0.3, 0.7]]
        mock_scaler.transform.return_value = [[1, 2, 3, 4, 5, 6, 7, 8, 9]]
        mock_encoders.__getitem__.return_value.transform.return_value = [0]
        
        payload = {
            "edad": 25,
            "genero": "Femenino",
            "estado_civil": "Soltero",
            "tipo_plan": "Basico",
            "meses_contrato": 3,
            "factura_mensual": 40.0,
            "GB_consumidos": 15.0,
            "llamadas_soporte": 5,
            "cambios_plan": 2
        }
        
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["churn"] == True
        assert data["churn_probability"] > 0.5

if __name__ == '__main__':
    pytest.main([__file__, '-v'])

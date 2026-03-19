"""FastAPI application for Churn Prediction."""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
from src.preprocessing import DataPreprocessor

app = FastAPI(
    title="Churn Prediction API",
    description="API REST para predecir la cancelación de clientes (churn) utilizando Machine Learning.",
    version="1.0.0",
)

models_path = os.path.join(os.path.dirname(__file__), "..", "models")
model = joblib.load(os.path.join(models_path, "churn_model.pkl"))
preprocessor = DataPreprocessor()
preprocessor.load_preprocessors(models_path)


class CustomerData(BaseModel):
    """Schema for customer data input."""
    
    edad: int = Field(..., ge=18, le=100, description="Edad del cliente (18-100)")
    genero: str = Field(..., description="Género (Masculino/Femenino)")
    estado_civil: str = Field(..., description="Estado civil")
    tipo_plan: str = Field(..., description="Tipo de plan (Basico/Estandar/Premium)")
    meses_contrato: int = Field(..., ge=0, description="Meses con la empresa")
    factura_mensual: float = Field(..., gt=0, description="Factura mensual ($)")
    GB_consumidos: float = Field(..., ge=0, description="GB consumidos")
    llamadas_soporte: int = Field(..., ge=0, description="Llamadas al soporte")
    cambios_plan: int = Field(..., ge=0, description="Cambios de plan")

    model_config = {
        "json_schema_extra": {
            "example": {
                "edad": 30,
                "genero": "Masculino",
                "estado_civil": "Soltero",
                "tipo_plan": "Premium",
                "meses_contrato": 12,
                "factura_mensual": 80.0,
                "GB_consumidos": 45.0,
                "llamadas_soporte": 1,
                "cambios_plan": 0,
            }
        }
    }


class PredictionResponse(BaseModel):
    """Schema for prediction response."""
    
    churn: bool
    churn_probability: float
    mensaje: str


@app.get("/", response_model=dict[str, str])
def root() -> dict[str, str]:
    """Root endpoint."""
    return {"message": "Churn Prediction API", "status": "running"}


@app.get("/health")
def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerData) -> PredictionResponse:
    """Predict customer churn."""
    try:
        data: dict[str, Any] = customer.model_dump()
        
        X = preprocessor.transform_input(data)
        
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0][1]
        
        mensaje = (
            "El cliente probablemente cancelará"
            if prediction
            else "El cliente probablemente permanecerá"
        )
        
        return PredictionResponse(
            churn=bool(prediction),
            churn_probability=round(float(probability), 4),
            mensaje=mensaje,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

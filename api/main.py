from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI(title="Churn Prediction API", description="API para predecir cancelación de clientes")

model = joblib.load('models/churn_model.pkl')
scaler = joblib.load('models/scaler.pkl')
label_encoders = joblib.load('models/label_encoders.pkl')

feature_names = ['edad', 'genero', 'estado_civil', 'tipo_plan', 'meses_contrato', 
                 'factura_mensual', 'GB_consumidos', 'llamadas_soporte', 'cambios_plan']

class CustomerData(BaseModel):
    edad: int
    genero: str
    estado_civil: str
    tipo_plan: str
    meses_contrato: int
    factura_mensual: float
    GB_consumidos: float
    llamadas_soporte: int
    cambios_plan: int

@app.get("/")
def root():
    return {"message": "Churn Prediction API", "status": "running"}

@app.post("/predict")
def predict_churn(customer: CustomerData):
    try:
        data = customer.model_dump()
        
        for col in ['genero', 'estado_civil', 'tipo_plan']:
            data[col] = label_encoders[col].transform([data[col]])[0]
        
        df = pd.DataFrame([data])[feature_names]
        df_scaled = scaler.transform(df)
        
        prediction = model.predict(df_scaled)[0]
        probability = model.predict_proba(df_scaled)[0][1]
        
        return {
            "churn": bool(prediction),
            "churn_probability": round(probability, 4),
            "mensaje": "El cliente probablemente cancelará" if prediction else "El cliente probablemente permanecerá"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy"}

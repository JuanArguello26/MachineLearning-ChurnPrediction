import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_data():
    df = pd.read_csv('data/churn_data.csv')
    return df

def prepare_data(df):
    df_model = df.drop('id_cliente', axis=1)
    df_model['churn'] = df_model['churn'].map({'Si': 1, 'No': 0})
    
    label_encoders = {}
    for col in ['genero', 'estado_civil', 'tipo_plan']:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col])
        label_encoders[col] = le
    
    X = df_model.drop('churn', axis=1)
    y = df_model['churn']
    
    return X, y, label_encoders

class TestDataLoading:
    def test_data_file_exists(self):
        assert os.path.exists('data/churn_data.csv'), "Data file not found"
    
    def test_data_not_empty(self):
        df = load_data()
        assert len(df) > 0, "Data is empty"
    
    def test_data_has_required_columns(self):
        df = load_data()
        required = ['id_cliente', 'edad', 'genero', 'estado_civil', 'tipo_plan', 
                    'meses_contrato', 'factura_mensual', 'GB_consumidos', 
                    'llamadas_soporte', 'cambios_plan', 'churn']
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

class TestDataPreparation:
    def test_prepare_data(self):
        df = load_data()
        X, y, encoders = prepare_data(df)
        assert X.shape[0] == len(y), "X and y length mismatch"
        assert len(encoders) == 3, "Should have 3 encoders"
    
    def test_churn_values(self):
        df = load_data()
        assert set(df['churn'].unique()) == {'Si', 'No'}, "Invalid churn values"

class TestModel:
    def test_model_training(self):
        df = load_data()
        X, y, _ = prepare_data(df)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_scaled, y)
        
        y_pred = model.predict(X_scaled)
        
        assert accuracy_score(y, y_pred) > 0.5, "Model accuracy too low"
    
    def test_model_predictions(self):
        df = load_data()
        X, y, _ = prepare_data(df)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_scaled, y)
        
        y_pred = model.predict(X_scaled[:5])
        y_proba = model.predict_proba(X_scaled[:5])
        
        assert len(y_pred) == 5, "Wrong prediction length"
        assert y_proba.shape == (5, 2), "Wrong probability shape"
        assert ((y_proba >= 0).all() and (y_proba <= 1).all()), "Probabilities out of range"

class TestAPIPayload:
    def test_customer_data_structure(self):
        sample = {
            'edad': 30,
            'genero': 'Masculino',
            'estado_civil': 'Soltero',
            'tipo_plan': 'Premium',
            'meses_contrato': 12,
            'factura_mensual': 50.0,
            'GB_consumidos': 30.0,
            'llamadas_soporte': 2,
            'cambios_plan': 0
        }
        
        required_keys = list(sample.keys())
        assert len(required_keys) == 9, "Wrong number of fields"

if __name__ == '__main__':
    pytest.main([__file__, '-v'])

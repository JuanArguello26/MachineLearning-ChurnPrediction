# Churn Prediction API

API REST para predecir la cancelación de clientes (churn) utilizando Machine Learning.

## Descripción del Proyecto

Este proyecto implementa un modelo de clasificación para predecir qué clientes probablemente cancelarán sus servicios. El modelo utiliza **Random Forest** y fue entrenado con datos de clientes de telecomunicaciones.

## Stack Tecnológico

- **Python 3.10+**
- **FastAPI** - Framework web
- **Scikit-learn** - Machine Learning
- **Pandas/NumPy** - Análisis de datos
- **Matplotlib/Seaborn** - Visualizaciones

## Estructura del Proyecto

```
churn-prediction/
├── api/
│   └── main.py          # API REST
├── data/
│   └── churn_data.csv   # Dataset
├── models/
│   └── *.pkl           # Modelos guardados
├── notebooks/
│   └── eda.ipynb       # Análisis exploratorio
├── tests/
│   ├── test_model.py   # Tests del modelo
│   └── test_api.py     # Tests de la API
├── requirements.txt
└── README.md
```

## Instalación

```bash
# Clonar el repositorio
git clone <repo-url>
cd churn-prediction

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate     # Windows

# Instalar dependencias
pip install -r requirements.txt
```

## Uso Local

### Ejecutar la API

```bash
cd api
uvicorn main:app --reload
```

La API estará disponible en: `http://localhost:8000`

### Documentación Interactive

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Ejemplo de Predicción

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "edad": 30,
    "genero": "Masculino",
    "estado_civil": "Soltero",
    "tipo_plan": "Premium",
    "meses_contrato": 12,
    "factura_mensual": 80.0,
    "GB_consumidos": 45.0,
    "llamadas_soporte": 1,
    "cambios_plan": 0
  }'
```

Respuesta esperada:
```json
{
  "churn": false,
  "churn_probability": 0.15,
  "mensaje": "El cliente probablemente permanecerá"
}
```

## Ejecutar Tests

```bash
pytest tests/ -v
```

## Deployment

### Railway

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app)

1. Conectar repositorio a Railway
2. Establecer `PYTHON_VERSION` = `3.10`
3. Deploy automático

### Render

1. Conectar repositorio a Render
2. Build Command: `pip install -r requirements.txt`
3. Start Command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`

## Métricas del Modelo

| Métrica | Valor |
|---------|-------|
| Accuracy | ~85% |
| Precision | ~80% |
| Recall | ~78% |
| F1-Score | ~79% |
| ROC-AUC | ~90% |

## Variables del Dataset

| Variable | Descripción |
|----------|-------------|
| edad | Edad del cliente |
| genero | Género (Masculino/Femenino) |
| estado_civil | Estado civil |
| tipo_plan | Tipo de plan (Basico/Estandar/Premium) |
| meses_contrato | Meses con la empresa |
| factura_mensual | Factura mensual ($) |
| GB_consumidos | GB consumidos |
| llamadas_soporte | Llamadas al soporte |
| cambios_plan | Cambios de plan |

## Licencia

MIT License

"""Configuration constants for churn prediction model."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Model configuration parameters."""
    
    n_estimators: int = 100
    max_depth: int = 10
    random_state: int = 42
    class_weight: str = "balanced"
    test_size: float = 0.2


MODEL_CONFIG = ModelConfig()

FEATURE_NAMES: list[str] = [
    "edad",
    "genero",
    "estado_civil",
    "tipo_plan",
    "meses_contrato",
    "factura_mensual",
    "GB_consumidos",
    "llamadas_soporte",
    "cambios_plan",
]

CATEGORICAL_COLUMNS: list[str] = ["genero", "estado_civil", "tipo_plan"]

TARGET_COLUMN: str = "churn"

CHURN_LABELS: dict[str, int] = {"Si": 1, "No": 0}

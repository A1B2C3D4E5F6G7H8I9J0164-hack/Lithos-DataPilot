"""
Pydantic schemas for FastAPI endpoints.
"""
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., json_schema_extra={"example": "healthy"})
    model_loaded: bool = Field(..., json_schema_extra={"example": True})
    model_name: Optional[str] = Field(None, json_schema_extra={"example": "XGBoost Classifier"})
    timestamp: str


class FeatureSchemaItem(BaseModel):
    name: str
    dtype: str
    sample_value: Any
    is_numerical: bool
    unique_values: Optional[List[Any]] = None


class ModelInfoResponse(BaseModel):
    status: str = Field(..., json_schema_extra={"example": "success"})
    best_model_name: str
    best_model_id: str
    problem_type: str
    target_column: str
    primary_metric: str
    best_score: float
    baseline_score: float
    feature_schema: List[FeatureSchemaItem]
    created_at: str


class MetricsResponse(BaseModel):
    status: str = Field(..., json_schema_extra={"example": "success"})
    primary_metric: str
    best_score: float
    baseline_score: float
    best_params: Dict[str, Any]
    problem_type: str


class SinglePredictionRequest(BaseModel):
    features: Dict[str, Any] = Field(
        ...,
        description="Feature dictionary mapping column names to values",
        json_schema_extra={"example": {"age": 45, "tenure_months": 24, "monthly_charges": 75.5, "contract_type": "One year"}}
    )


class SinglePredictionResponse(BaseModel):
    status: str = Field(..., json_schema_extra={"example": "success"})
    prediction: Union[int, float, str]
    probability: Optional[float] = None
    probabilities: Optional[Dict[str, float]] = None
    problem_type: str


class BatchPredictionRequest(BaseModel):
    records: List[Dict[str, Any]] = Field(
        ...,
        description="List of feature dictionaries",
    )


class BatchPredictionResponse(BaseModel):
    status: str = Field(..., json_schema_extra={"example": "success"})
    count: int
    predictions: List[Union[int, float, str]]
    probabilities: Optional[List[Optional[float]]] = None

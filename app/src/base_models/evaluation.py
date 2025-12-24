from pydantic import BaseModel
from typing import TypeAlias

class EvaluationResult(BaseModel):
    rmse: float
    mae: float
    mape: float
    r2: float

EvaluationResults: TypeAlias = list[EvaluationResult]
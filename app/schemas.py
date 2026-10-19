
from pydantic import BaseModel, Field, ConfigDict


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: list[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description=(
            "Four Iris measurements: sepal length, sepal width, "
            "petal length, and petal width (cm)."
        )
    )


class PredictResponse(BaseModel):
    prediction: str
    probabilities: dict[str, float]
    model_version: str

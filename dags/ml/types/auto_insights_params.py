from pydantic import BaseModel, validator


class AutoInsightsParams(BaseModel):
    insight_type: str
    qmin: float
    qmax: float
    n_partitions: int

    @validator("qmin")
    def qmin_from_0_to_1(cls, v):
        assert 0 <= v <= 1, (
            "qmin must be from 0 to 1"
        )
        return v

    @validator("qmax")
    def qmax_from_0_to_1(cls, v):
        assert 0 <= v <= 1, (
            "qmax must be from 0 to 1"
        )
        return v

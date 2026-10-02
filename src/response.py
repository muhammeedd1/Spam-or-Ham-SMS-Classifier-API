from pydantic import BaseModel , Field , field_validator
from typing import Literal


class SMSResponse(BaseModel) :
    label : Literal["spam" , "ham"]
    is_spam : bool
    confidence: float = Field(ge=0.0, le=1.0)


class SMSBatchResponse(BaseModel) :
    results : list[SMSResponse]


class HealthResponse(BaseModel):
    status : str
    model_loaded : bool
    app_version : str


class ErrorResponse(BaseModel) :
    detail : str
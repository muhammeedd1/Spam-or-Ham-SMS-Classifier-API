import logging

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader

from src.config import settings
from src.inference import spamClassifier
from src.request import SMSRequest, BatchSMSRequest
from src.response import (
    SMSBatchResponse,
    SMSResponse,
    ErrorResponse,
    HealthResponse,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("spam_ham_api")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


api_key_header = APIKeyHeader(name="x-api-key")


async def verify_api_key(
    api_key: str = Depends(api_key_header)
):
    if api_key != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not authorized to use this API",
        )

    return api_key


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Meta"],
    summary="Health Check",
)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=spamClassifier.model is not None,
        app_version=settings.APP_VERSION,
    )


@app.post(
    "/predict",
    response_model=SMSResponse,
    responses={
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    tags=["Classification"],
    summary="Classify a single SMS",
)
def predict(
    payload: SMSRequest,
    api_key: str = Depends(verify_api_key),
) -> SMSResponse:

    try:
        result = spamClassifier.predict(payload.text)

        return SMSResponse(**result)

    except Exception:

        logger.exception(
            "Prediction failed for input of length %d",
            len(payload.text),
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to classify the provided text.",
        )


@app.post(
    "/predict/batch",
    response_model=SMSBatchResponse,
    responses={
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    tags=["Classification"],
    summary="Classify multiple SMS messages",
)
def predict_batch(
    payload: BatchSMSRequest,
    api_key: str = Depends(verify_api_key),
) -> SMSBatchResponse:

    try:
        results = [
            spamClassifier.predict(text)
            for text in payload.texts
        ]

        return SMSBatchResponse(
            results=[
                SMSResponse(**result)
                for result in results
            ]
        )

    except Exception:

        logger.exception(
            "Batch prediction failed for %d texts",
            len(payload.texts),
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to classify one or more of the provided texts.",
        )


@app.get(
    "/",
    tags=["Meta"],
    summary="API root",
)
def root() -> dict:
    return {
        "message": f"{settings.APP_NAME} is running.",
        "docs": "/docs",
        "health": "/health",
    }
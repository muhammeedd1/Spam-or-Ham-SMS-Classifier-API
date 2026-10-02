from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):

    APP_NAME: str = "Spam or Ham SMS Classifier API"
    APP_VERSION: str = "1.0.0"

    API_KEY: str

    MODEL_PATH: Path = BASE_DIR / "models" / "Best_model.pkl"
    TFIDF_VECTORIZER_PATH: Path = BASE_DIR / "models" / "tfidf_vectorizer.pkl"
    LABEL_ENCODER_PATH: Path = BASE_DIR / "models" / "Label_encoder.pkl"

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
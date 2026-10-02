FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NLTK_DATA=/usr/local/share/nltk_data

WORKDIR /app

# Install Python dependencies first for better Docker layer caching
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt \
    && python -m nltk.downloader \
        -d /usr/local/share/nltk_data \
        punkt \
        punkt_tab \
        stopwords \
        wordnet \
        omw-1.4

# Copy only runtime files
COPY src/ ./src/
COPY models/ ./models/

# Create a non-root user
RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app /usr/local/share/nltk_data

USER appuser

# Render provides PORT; 10000 is the local/default fallback
ENV PORT=10000

EXPOSE 10000

CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
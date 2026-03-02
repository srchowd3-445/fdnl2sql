FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip && \
    pip install \
      fastapi \
      uvicorn[standard] \
      openai \
      pydantic \
      numpy \
      sentence-transformers

EXPOSE 8080

CMD ["sh", "-c", "uvicorn chat_pipeline_api:app --host 0.0.0.0 --port ${PORT:-8080}"]

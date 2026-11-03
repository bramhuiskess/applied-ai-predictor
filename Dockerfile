# Stage 1: generate the demo data and train every use case.
FROM python:3.12-slim AS train
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml ./
COPY predictor ./predictor
RUN pip install --no-cache-dir ".[train]"
COPY scripts ./scripts
COPY usecases ./usecases
COPY data ./data
RUN python scripts/make_datasets.py && predictor train

# Stage 2: the serving image. No pandas, scikit-learn or LightGBM: the core only
# needs the ONNX runtime, which proves the core is independent of training libraries.
FROM python:3.12-slim AS serve
WORKDIR /app
COPY pyproject.toml ./
COPY predictor ./predictor
RUN pip install --no-cache-dir . && useradd --create-home predictor
COPY usecases ./usecases
COPY --from=train /app/models ./models
USER predictor
ENV PREDICTOR_USECASES_DIR=/app/usecases PREDICTOR_MODELS_DIR=/app/models
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["predictor", "serve", "--host", "0.0.0.0", "--port", "8000"]

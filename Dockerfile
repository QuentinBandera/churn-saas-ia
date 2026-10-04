FROM python:3.12-slim

RUN useradd -m -u 10001 appuser
WORKDIR /app

COPY api/requirements.txt ./api/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt

COPY api/ ./api/
COPY models/churn_model_v1.0.0.joblib models/churn_model_v1.0.0.json ./models/
COPY catalogue_plans.csv ./

ENV PYTHONUNBUFFERED=1 CHURN_MODEL_DIR=/app/models CHURN_MODEL_VERSION=1.0.0
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD python -c "import urllib.request,sys;sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]

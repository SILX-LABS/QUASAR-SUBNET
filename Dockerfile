FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV QUASAR_DASHBOARD_HOST=0.0.0.0
ENV QUASAR_NETWORK=finney
ENV QUASAR_NETUID=24
ENV QUASAR_S3_BUCKET=quasar-incentive-sn24-529337356998-us-east-1
ENV QUASAR_S3_REGION=us-east-1
ENV QUASAR_S3_ANONYMOUS=true
ENV QUASAR_DASHBOARD_PUBLISH_S3=false
ENV QUASAR_DASHBOARD_RATE_LIMIT_REQUESTS=120
ENV QUASAR_DASHBOARD_RATE_LIMIT_WINDOW_SEC=60
ENV QUASAR_DASHBOARD_CORS_ORIGINS=*

WORKDIR /app

COPY pyproject.toml README.md LICENSE /app/
COPY incentive /app/incentive
COPY frontend /app/frontend
COPY scripts/run_dashboard_backend.py /app/scripts/run_dashboard_backend.py

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -e .

CMD ["quasar-dashboard-backend"]

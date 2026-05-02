FROM python:3.11-slim-bullseye

LABEL maintainer="WikiRisk Team"
LABEL description="MLflow Tracking Server for WikiRisk"

RUN pip install --no-cache-dir mlflow==2.13.2 boto3==1.34.84

RUN useradd -m -u 1001 mlflow
RUN mkdir -p /mlflow && chown mlflow:mlflow /mlflow
USER mlflow
WORKDIR /mlflow

EXPOSE 5001

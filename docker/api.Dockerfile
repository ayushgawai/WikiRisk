# ─────────────────────────────────────────────────────────────────────────────
# WikiRisk – API/Spark Dockerfile
# Used by: api, streaming, collector services
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim-bullseye AS base

# Java is required for PySpark
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYSPARK_PYTHON=python3

RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jre-headless \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Dependencies ──────────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Source code ───────────────────────────────────────────────────────────────
COPY src/ ./src/
COPY manifests/ ./manifests/

# ── Non-root user ─────────────────────────────────────────────────────────────
RUN useradd -m -u 1001 appuser && \
    mkdir -p /app/data && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "src.serving.main:app", "--host", "0.0.0.0", "--port", "8000"]

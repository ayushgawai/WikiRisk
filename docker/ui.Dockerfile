# ─────────────────────────────────────────────────────────────────────────────
# WikiRisk – Streamlit UI Dockerfile
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim-bullseye

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install only UI dependencies (no PySpark/Java needed)
RUN pip install --no-cache-dir \
    streamlit==1.35.0 \
    plotly==5.22.0 \
    altair==5.3.0 \
    httpx==0.27.0 \
    python-dotenv==1.0.1

COPY src/ui/ ./src/ui/

RUN useradd -m -u 1001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "src/ui/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--theme.base=light", \
     "--server.headless=true"]

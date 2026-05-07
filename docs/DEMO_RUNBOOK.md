# WikiRisk Demo Runbook

## Project Framing

WikiRisk is a production-style big-data demo for real-time Wikipedia edit risk
monitoring. The point is to show the full lifecycle:

1. Historical Wikipedia revisions are processed with Spark batch jobs.
2. Weak labels and SparkML features train a Logistic Regression classifier.
3. Training metadata, metrics, and artifacts are tracked in MLflow.
4. Wikimedia EventStreams provide live edit events.
5. Spark Structured Streaming scores those events with the saved model.
6. SQLite stores the live prediction feed for fast API and UI access.
7. FastAPI serves health, stats, edits, explanations, and notifications.
8. Streamlit visualizes the live system for moderators, watched-page owners,
   and demo audiences.

## Start The Demo

Terminal 1:

```bash
make mlflow
```

Terminal 2:

```bash
make serve
```

Terminal 3:

```bash
make ui
```

Terminal 4:

```bash
make live
```

Open:

- Dashboard: http://localhost:8501
- API docs: http://localhost:8000/docs
- MLflow: http://localhost:5001

## Presentation Flow

1. Show the dashboard metrics and risk distribution.
2. Filter to HIGH or MEDIUM risk edits.
3. Select one edit and open its detail panel.
4. Generate an explanation.
5. Click Send alert. If secrets are not configured, show the setup message.
6. Switch to API docs and show `/stats`, `/edits/recent`, and `/explain`.
7. Switch to MLflow and show experiment runs, metrics, params, and artifacts.
8. Briefly show `data/wikirisk.db` or the schema slide as the serving store.

## Slack Alert Setup

Set this in `.env`:

```bash
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
```

Then restart the API and dashboard. The dashboard's Send alert button will post
the selected edit to the configured Slack channel.

## Email Alert Setup

For Gmail with an app password:

```bash
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your.email@gmail.com
SMTP_PASSWORD=your_16_character_app_password
SMTP_FROM_EMAIL=your.email@gmail.com
ALERT_TO_EMAIL=recipient@example.com
```

For Resend SMTP:

```bash
SMTP_HOST=smtp.resend.com
SMTP_PORT=587
SMTP_USERNAME=resend
SMTP_PASSWORD=your_resend_api_key
SMTP_FROM_EMAIL=alerts@your_verified_domain.com
ALERT_TO_EMAIL=recipient@example.com
```

Restart the API after changing `.env`.

## What To Say If Asked About MLflow

MLflow is not in the low-latency serving path. It is used for reproducibility,
experiment tracking, model provenance, metrics, params, and artifacts. The
streaming processor loads the saved model artifact for live scoring, while the
MLflow UI proves the training lifecycle happened and can be reproduced.

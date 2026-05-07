## Gemini Prompt: Generate a WikiRisk Architecture Diagram

Create a polished, presentation-ready architecture diagram for a project named
**WikiRisk — Real-Time Wikipedia Edit Risk Detection System**.

Style requirements:
- clean, modern, professional data-engineering aesthetic
- use a Wikipedia-inspired color palette: off-white paper background, deep blue
  accents, dark gray text, subtle gold highlights
- make it look like a serious university capstone / production demo diagram
- include icons or visual cues for batch processing, streaming, API serving,
  SQLite storage, and ML model tracking
- avoid clutter; keep the diagram readable on a slide
- use a left-to-right flow with clear stage labels
- make the diagram feel like a big-data platform rather than a simple toy app

Diagram content to include:
1. Historical data source block
   - English Wikipedia Dumps
   - MediaWiki revision history
   - large-scale batch ingestion

2. Batch pipeline block
   - Spark batch parsing and feature engineering
   - weak-label generation
   - SparkML training
   - MLflow experiment tracking
   - model registry / saved model artifact

3. Streaming pipeline block
   - Wikimedia EventStreams / RecentChange API
   - Spark Structured Streaming
   - real-time feature extraction
   - model scoring
   - risk label assignment

4. Storage and serving block
   - SQLite / local prediction store
   - Parquet / processed data archive
   - FastAPI backend
   - health, stats, recent edits, explain, and notify endpoints
   - Streamlit dashboard

5. Output / demo impact block
   - moderation triage
   - misinformation early warning
   - integrity monitoring
   - real-time trust scoring
   - Slack/email alerting for watched pages

Suggested layout:
- top row: data sources
- middle row: batch training pipeline and streaming scoring pipeline
- lower row: storage and API serving
- final row: end-user impact / applications

Text to include on the diagram:
- WikiRisk
- Real-Time Wikipedia Edit Risk Detection System
- Spark Batch Processing
- Spark Structured Streaming
- SparkML Logistic Regression
- MLflow Tracking
- FastAPI Serving Layer
- Streamlit Live Dashboard
- Slack / Email Alerts

Extra notes:
- show that the model is already trained and reused for live scoring
- emphasize that this is a production-style demo, not a toy notebook
- include arrows that show data flowing from historical training to live inference
- include the Streamlit dashboard as the live visualization layer that reads from FastAPI
- show MLflow UI as a separate tracking/ops tool, not as part of the
   serving path

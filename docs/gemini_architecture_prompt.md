## Gemini Prompt: Generate a Minimal, Professional WikiRisk Architecture Diagram

Create a single-slide, presentation-ready architecture diagram for **WikiRisk — Real-Time Wikipedia Edit Risk Detection System**. Keep the design clean, minimalist, and academic-professional.

Design brief (very short):
- Visual style: minimalist, lucid lines, generous whitespace, Wikipedia-inspired palette (off-white paper, deep blue accents, dark gray text). Use a small, tasteful WikiRisk wordmark or generic project logo at the top-left.
- Layout: left-to-right data flow with 3 tiers (Data → Processing → Serving) and a small Ops/Tracking box off to the right.
- Iconography: use simple icons for (dump/archive), (stream/event), (Spark/batch), (streaming worker), (model artifact), (DB), (API), (dashboard/client), (alerts).
- Typography: short, bold labels; avoid paragraph text.

Content blocks and connections (explicit):
- Data sources (left): "Wikipedia dumps" and "Wikimedia RecentChange (EventStreams)". Arrow from dumps → batch pipeline; arrow from EventStreams → streaming pipeline.
- Batch training (center-top): "Spark batch processing → feature engineering → training (SparkML)". Show saved model artifact (label: "Saved PipelineModel / artifact") with arrow pointing to the Serving/Scoring component.
- Streaming scoring (center-bottom): "Streaming collector → real-time feature extraction → model scoring (uses saved model) → risk label assignment". Note: streaming scorer can be Spark Structured Streaming or a lightweight demo scorer; show this as interchangeable with a small label "(Spark or demo scorer)".
- Serving & storage (right): "FastAPI" box that reads/writes the local SQLite prediction store and exposes endpoints: `/health`, `/stats`, `/edits/recent`, `/edits/{id}`, `/explain` (AI), `/notify` (alerts). Show Parquet / processed archive for historical data.
- Ops / Tracking (top-right, separate): "MLflow (tracking & model registry)" with arrows from Batch training → MLflow (tracking) and from explain/API → MLflow (explanation provenance). Mark MLflow as read-only to serving path.
- Endpoints / consumers (far-right): small icons for "Moderator Dashboard (presenter)" and "Slack / Email alerts" connected to FastAPI.

Text labels to place sparingly: WikiRisk; Real-time edit risk scoring; Saved model reused for inference; SQLite prediction store; MLflow (tracking & artifacts); Streaming collector (EventStreams).

Deliverable instructions for generator:
- Produce a single SVG or PNG suitable for a slide (16:9) with clear vector shapes.
- Provide layered output or clear grouping so elements can be moved on a slide.
- Use arrows with directional heads; annotate only where helpful (no long notes).
- Emphasize: the model is pre-trained; live scoring re-uses the saved artifact; MLflow is for provenance, not in the request path.

Do NOT include Streamlit as a required component — the presenter UI reads from FastAPI. Keep the diagram uncluttered and suitable for a professor-review slide.

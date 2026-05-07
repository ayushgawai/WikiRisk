# Presentation Deck Prompt: WikiRisk (Slide-by-Slide)

Create a concise, professional presentation deck for the professor-review sequence below. Keep each slide minimalist, with 3–5 bullets, short speaker notes, and a single visual suggestion. Use the Gemini architecture diagram on the System Architecture slide.

Project title: **WikiRisk — Real-Time Wikipedia Edit Risk Detection System**

Presentation Sequence (must follow exactly):
1. Problem Statement
2. Historical / Realtime Dataset
3. System Architecture
4. DB Schema
5. Implementation Details
6. Visualization / Live Demo (must be live)
7. Conclusion (summary, lessons, future work)

Constraints and tone:
- The system is already trained and running locally — do NOT describe retraining.
- Emphasize production-style provenance: saved model artifact, streaming collector, FastAPI serving, SQLite prediction store, MLflow tracking.
- Clean, academic-professional visuals; avoid dense paragraphs and unnecessary jargon.
- Slides must be speaker-friendly: include short speaker notes (1–2 lines) per slide.

Slide-by-slide outline (deliverable):
- For each slide produce: `title`, `3–5 bullets`, `speaker notes` (1–2 lines), `visual suggestion` (one short line).

Specific guidance per slide:

1) Problem Statement
- Bullets: (a) high-volume collaborative editing leads to risky/harmful edits appearing quickly; (b) human review is slow and costly; (c) need lightweight automated triage with explainability.
- Speaker note: one sentence framing impact and motivation.
- Visual: single contextual image / icon showing Wikipedia edits flowing in.

2) Historical / Realtime Dataset
- Bullets: (a) historical dumps (Wikimedia revision history) for training; (b) EventStreams RecentChange for live events; (c) key schema and volume characteristics.
- Speaker note: mention sampling strategy and what fields are used (title, summary, byte delta, anon flag).
- Visual: two small boxes labeled "historical dumps" and "EventStreams" with brief stats.

3) System Architecture
- Bullets: (a) batch training produces a saved PipelineModel (SparkML) tracked in MLflow; (b) streaming collector + real-time scorer reuse saved model; (c) FastAPI exposes endpoints and reads/writes the SQLite prediction store; (d) MLflow provides provenance and artifacts (separate ops UI).
- Speaker note: reference the Gemini-generated diagram and highlight the pre-trained model reuse.
- Visual: place the Gemini SVG diagram centered on the slide.

4) DB Schema
- Bullets: (a) SQLite prediction store is the demo's source of truth; (b) core fields: id, rev_id, page_title, namespace, user, is_anon, comment, length_delta, timestamp, wiki, risk_score, risk_label, scored, created_at; (c) used by API and visualizer for fast retrieval.
- Speaker note: note simple indexing for recent retrieval and filters by risk_label.
- Visual: compact table showing the core fields (no raw data).

5) Implementation Details
- Bullets: (a) Spark for batch feature engineering and model training (PipelineModel); (b) streaming scorer (Spark Structured Streaming or demo scorer) for low-latency scoring; (c) FastAPI serving layer with `/health`, `/stats`, `/edits`, `/explain`; (d) MLflow used for experiment tracking and explanation provenance.
- Speaker note: mention language/runtime (Python, PySpark), and saved artifact locations (mlruns/, data/).
- Visual: small flow diagram or code snippet callout (API call example).

6) Visualization / Live Demo (must be live)
- Bullets: (a) live dashboard reads from FastAPI and filters by risk; (b) explain endpoint produces LLM or rule-based explanations (logged to MLflow); (c) demo shows new edits appearing, scoring, and MLflow runs for explanations; (d) alerts visible via Slack/email integration.
- Speaker note: instruct presenter to run the supervisor (collector + demo scorer) before demo and open MLflow UI and the dashboard.
- Visual: screenshot placeholders + note: "Open MLflow UI and API console during demo." 

7) Conclusion
- Bullets: (a) summary of system capabilities; (b) lessons learned (streaming complexity, ML explainability trade-offs); (c) future work (scale Spark processor, add role-based alerts, improve human feedback loop).
- Speaker note: one-line closing call-to-action.
- Visual: 3 bullet takeaway icons.

Opening script (2–3 lines):
"Today I'll show WikiRisk — a lightweight, production-style system that detects and explains risky Wikipedia edits in real time. We'll walk datasets, architecture, and a live demo that connects streaming edits to ML provenance and explainability."

Closing script (2–3 lines):
"In summary: WikiRisk demonstrates how saved ML artifacts, streaming collection, and lightweight serving can provide practical, explainable triage for Wikipedia moderation. Next steps focus on scale and operational hardening. I'm happy to take questions."

Deliverable format reminder: output a slide-by-slide outline exactly as specified (title, bullets, speaker notes, visual suggestion). Keep wording concise and slide-ready.

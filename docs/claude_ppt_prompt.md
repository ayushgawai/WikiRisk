# Claude Prompt: Create the WikiRisk Presentation Deck

Create a polished, professional presentation deck for a university big-data project.

Project title: **WikiRisk: Real-Time Wikipedia Edit Risk Detection System**

Use the following presentation sequence exactly:

1. Problem Statement
2. Historical / Realtime Dataset
3. System Architecture
4. DB Schema
5. Implementation Details
6. Visualization / Live Demo (must be live)
7. Conclusion (summary, lessons, future work)

Important constraints:
- The project is already trained and running locally. Do **not** describe retraining as part of the demo.
- Emphasize that the system uses the saved model, streaming pipeline, API, and SQLite prediction store that are already built.
- The deck should feel like a real production big-data demo, not a toy notebook.
- Make it visually clean, modern, and easy to present aloud.
- Include concise speaker-friendly bullets, not long paragraphs.
- The live demo should focus on the dashboard, API, streaming output, MLflow tracking, and saved prediction store.
- The architecture slide should incorporate the Gemini-generated architecture diagram.

What the deck should communicate:
- Wikipedia is a high-volume collaborative platform where harmful edits can appear in real time.
- WikiRisk uses Spark batch processing on historical Wikipedia revision data.
- A SparkML classifier scores edit risk using saved artifacts and MLflow-tracked training results.
- Spark Structured Streaming handles incoming recent-change events.
- FastAPI serves health, stats, recent-edit, explanation, and notification data.
- The serving layer exposes health, stats, recent-edit, explain, and notify endpoints for the demo.
- The database stores predictions and live records for fast retrieval.
- The Streamlit dashboard is the presenter-facing visualization layer.

Risk calculation guidance:
- Explain that the model outputs `risk_score`: the SparkML Logistic Regression probability for class 1, trained from historical Wikimedia `revision_is_identity_reverted` labels.
- Explain that class 1 means the edit was later identity-reverted in Wikipedia history, so the score estimates "likelihood this edit resembles edits the community later reverted."
- Mention the main features: TF-IDF edit summary/comment, TF-IDF page title, byte delta, anonymous editor flag, namespace, hour of day, and day of week.
- Explain the live risk type mapping from `src/config/settings.py`: HIGH when score >= 0.20, MEDIUM when score >= 0.08, otherwise LOW.
- Make clear that OpenAI explanations do not calculate the risk; they explain the already-computed SparkML score.

DB schema guidance:
- Show the SQLite prediction store as the source of truth for the demo.
- Include the core fields: id, rev_id, page_title, namespace, user, is_anon, comment, length_delta, timestamp, wiki, risk_score, risk_label, scored, created_at.
- Mention that the database is populated by the streaming pipeline and read by the API/UI.

Visualization / live demo guidance:
- Show the actual workflow end to end through the Streamlit dashboard.
- Demonstrate live risk filters, edit detail inspection, AI/rule-based explanation generation, alert trigger, API responses, the SQLite prediction store, the streaming processor, and MLflow runs/artifacts.
- Mention that the demo is based on the already-trained model and saved artifacts.
- Make clear that the UI is a thin visualization and operations layer over FastAPI and SQLite, not a notebook.

Style guidance:
- Use a professional academic / startup-demo tone.
- Prefer clear section headers and slide titles.
- Include 1-2 key takeaway bullets per slide.
- Suggest simple visual layouts when helpful.

Output format:
- Provide a slide-by-slide outline.
- For each slide, include:
  - slide title
  - 3-5 bullets
  - speaker notes when useful
  - visual suggestion when useful

Also include a short opening and closing script the presenter can read aloud.

Do not add unnecessary technical jargon. Keep it sharp, credible, and presentation-ready.

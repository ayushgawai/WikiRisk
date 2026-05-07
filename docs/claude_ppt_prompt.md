# Claude Prompt: Create the WikiRisk Presentation Deck

Create a polished, professional presentation deck for a university big-data project.

Project title: **WikiRisk: Real-Time Wikipedia Edit Risk Detection System**

Use the following presentation sequence exactly:

1. Problem Statement
2. Historical / Realtime Dataset
3. System Architecture
4. DB Schema
5. Implementation Details
6. Live Demo Flow (must be live)
7. Conclusion (summary, lessons, future work)

Important constraints:
- The project is already trained and running locally. Do **not** describe retraining as part of the demo.
- Emphasize that the system uses the saved model, streaming pipeline, and dashboard that are already built.
- The deck should feel like a real production big-data demo, not a toy notebook.
- Make it visually clean, modern, and easy to present aloud.
- Include concise speaker-friendly bullets, not long paragraphs.
- The live demo should focus on the API, streaming output, and saved prediction store.
- The architecture slide should incorporate the Gemini-generated architecture diagram.

What the deck should communicate:
- Wikipedia is a high-volume collaborative platform where harmful edits can appear in real time.
- WikiRisk uses Spark batch processing on historical Wikipedia revision data.
- A SparkML classifier scores edit risk using saved artifacts and MLflow-tracked training results.
- Spark Structured Streaming handles incoming recent-change events.
- FastAPI serves health, stats, and recent-edit data.
- The serving layer exposes health, stats, and recent-edit endpoints for the demo.
- The database stores predictions and live records for fast retrieval.

DB schema guidance:
- Show the SQLite prediction store as the source of truth for the demo.
- Include the core fields: id, rev_id, page_title, namespace, user, is_anon, comment, length_delta, timestamp, wiki, risk_score, risk_label, scored, created_at.
- Mention that the database is populated by the streaming pipeline and read by the API/UI.

Visualization / live demo guidance:
- Show the actual backend workflow end to end.
- Demonstrate the API responses, the SQLite prediction store, and the streaming processor.
- Mention that the demo is based on the already-trained model and saved artifacts.

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
  - optional speaker notes
  - optional visual suggestion

Also include a short opening and closing script the presenter can read aloud.

Do not add unnecessary technical jargon. Keep it sharp, credible, and presentation-ready.
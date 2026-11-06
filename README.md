# Multi-Format Creative Writing Continuity Engine

A RAG-powered continuity verification system for creative writing. Uses Google Gemini and ChromaDB to detect plot holes, timeline errors, prop inconsistencies, and structural gaps across three domains: **novels**, **screenplays**, and **standup comedy**.

---

## Domains

### 📖 Narrative (Novels)
- **Character state tracking:** Extracts character locations, inventory changes, and physical condition across chapters.
- **RAG context retrieval:** Chunks and indexes prior chapters with ChromaDB and `all-MiniLM-L6-v2` embeddings.
- **Contradiction detection:** Flags contradictions in new chapter drafts against accumulated canon and retrieved context.
- **Relationship graph:** Tracks directional relationships and their lifecycle (`active`, `superseded`, `ended`).

### 🎬 Screenplay (Scenes)
- **Props and character physical state:** Tracks props (`in_use`, `offscreen`, `discarded`, `destroyed`) and character health/location per scene.
- **Timeline consistency:** Checks for teleportation, timeline errors, and inconsistent injuries across scenes.
- **Character-prop graph:** Visualizes character-prop associations via interactive graph.

### 🎤 Standup Comedy (Sets)
- **Joke classification:** Classifies units as premises (setups), punchlines (payoffs), tags (extensions), or callbacks.
- **Deterministic coverage check:** Pure Python logic (no LLM calls) that flags abandoned setups and premises without punchlines.
- **Laugh density metrics:** Computes per-premise and overall laugh densities.

---

## Project Structure

```
├── app/
│   └── streamlit_app.py          # Interactive dashboard with 3-domain switcher
├── core/
│   ├── llm_config.py             # Gemini configuration & model fallback cascade
│   ├── schema.py                 # Pydantic models for narrative lore
│   ├── extraction.py             # Character state extraction
│   ├── rag.py                    # ChromaDB ingestion & retrieval pipeline
│   ├── verification.py           # Narrative contradiction checking
│   ├── relationships.py          # Relationship extraction and graph state
│   ├── graph_builder.py          # Streamlit-agraph visualization adapters
│   ├── screenplay_schema.py      # Pydantic models for screenplays
│   ├── screenplay_extraction.py  # Screenplay scene extraction
│   ├── screenplay_verification.py# Scene continuity verification
│   ├── standup_schema.py         # Pydantic models for comedy sets
│   ├── standup_extraction.py     # 4-way joke classification chain
│   └── standup_coverage.py       # Deterministic coverage & laugh density
├── data/
│   ├── sample_chapters/          # 4 sample fantasy chapters
│   ├── sample_screenplay/        # 5 sample screenplay scenes
│   └── sample_standup/           # Sample comedy set transcript
├── eval/
│   ├── test_cases*.json          # Test cases and ground truth
│   ├── run_eval.py               # Narrative evaluation harness
│   ├── run_eval_standup.py       # Standup evaluation harness
│   └── test_relationships.py     # Relationship extraction test
├── requirements.txt
└── .env.example
```

---

## Getting Started

### 1. Installation

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configuration

Copy `.env.example` to `.env` and add your Gemini API key:

```bash
cp .env.example .env
```

```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 3. Run the Streamlit Dashboard

```bash
streamlit run app/streamlit_app.py
```

Use the domain selector in the sidebar to toggle between **Narrative**, **Screenplay**, and **Standup Comedy**.

---

## Running Verification Scripts & Evals

```bash
# Narrative
python core/test_extraction_manual.py      # Sequential chapter extraction
python core/test_rag_manual.py             # Semantic retrieval tests
python eval/run_eval.py                    # Evaluation harness (recall / FPR)
python eval/test_relationships.py          # Relationship tracking test

# Screenplay
python core/test_screenplay_manual.py      # Scene extraction test

# Standup Comedy
python core/test_standup_manual.py         # Segment extraction & coverage test
python eval/run_eval_standup.py            # Coverage & classification eval
```

---

## Tech Stack

- **LLM:** Google Gemini (`gemini-2.5-flash`, `gemini-3.6-flash`, `gemini-2.5-pro` cascade)
- **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Vector Store:** ChromaDB
- **Schemas & Validation:** Pydantic v2
- **Orchestration:** LangChain
- **Frontend:** Streamlit & streamlit-agraph


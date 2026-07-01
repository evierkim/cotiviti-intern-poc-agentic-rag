# Agentic RAG for Risk Adjustment Decision Support

---

## What This Is

The **RAG-powered coder decision support** for risk adjustment ingests unstructured clinical notes, retrieves the most relevant evidence with semantic search, and uses an LLM to help medical coders reason about ICD-10 capture - with transparent retrieval, confidence scoring, and human-in-the-loop guardrails.

Scoped for speed and clarity: proves first principles, not a production platform.

## Why It Matters for Cotiviti

| Essay theme | How this PoC demonstrates it |
|---|---|
| Clinical NLP on unstructured notes | Chunks and indexes physician-style chart text |
| LLM-enhanced coding workflows | GPT-3.5 generates grounded coding recommendations |
| Agentic AI direction | Retrieve -> generate -> self-evaluate -> suggest next actions |
| Transparency / black-box concerns | Shows retrieved source chunks and similarity scores |
| Risk adjustment & medical record coding | Sample scenarios mirror HCC-relevant comorbidities |

## Architecture

```
Clinical Notes (.txt / .pdf)
        |
        v
 DocumentProcessor -> chunk + extract entities (ICD-10, meds, labs)
        |
        v
 VectorStore (Sentence-BERT + FAISS)
        |
        v
 RAGAgent -> retrieve top-k chunks
        |      generate answer (OpenAI)
        |      score confidence from retrieval
        +----> suggest coder next actions
        |
        v
 Gradio UI (interactive demo)
```

**Stack:** Python 3.10-3.12 | Gradio | OpenAI GPT-3.5-Turbo | Sentence-BERT (`all-MiniLM-L6-v2`) | FAISS

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/yourusername/cotiviti-intern-poc-agentic-rag.git
cd cotiviti-intern-poc-agentic-rag

python -m venv venv
venv\Scripts\activate # --> Windows
# source venv/bin/activate  --> macOS/Linux

pip install -r requirements.txt
```

### 2. Configure API key

```bash
copy .env.example .env # --> Windows
# cp .env.example .env # --> macOS/Linux
```

Add your OpenAI key to `.env`:

```
OPENAI_API_KEY=sk-...
```

### 3. Launch the interactive demo

```bash
python app/gradio_app.py
```

Open the local URL Gradio prints (typically `http://127.0.0.1:7860`).

### 4. Optional: run headless demo

Useful for verifying the pipeline without the UI:

```bash
python scripts/run_demo.py
```

## Demo Walkthrough (~2 minutes)

1. **Load Sample Notes** - indexes five synthetic clinical charts from `sample_notes.txt`
2. **Review Extracted Entities** - ICD-10 codes, medications, and labs detected via regex (NLP baseline layer)
3. **Pick an example question** or type your own, e.g.:
   - *What ICD-10 codes apply to the diabetic patient with hypertension and elevated A1C?*
   - *Does the COPD patient documentation support tobacco-use coding for risk adjustment?*
4. **Run Agentic Workflow** - review the coder answer, confidence label, and suggested next actions
5. **Expand "Retrieved Source Chunks"** - audit trail showing exactly which text grounded the response

## Project Structure

```
app/
  gradio_app.py          # Interactive UI
scripts/
  run_demo.py            # Headless CLI demo
src/
  clinical_notes_loader.py
  document_processor.py  # Chunking, PDF parsing, entity extraction
  vector_store.py        # Sentence-BERT + FAISS
  rag_agent.py           # Agentic RAG workflow
sample_notes.txt         # Synthetic clinical charts for demo
requirements.txt
.env.example
```

## Design Notes

- **No fine-tuning**. Prompt-engineered general-purpose LLM, matching the essay's "shift to general-purpose models" trend
- **Regex entity extraction** . Lightweight baseline NLP, surfaced for transparency (not hidden inside the LLM)
- **In-memory FAISS index**. No database overhead; rebuilds on each load for demo simplicity
- **Human-in-the-loop**. Every response includes suggested review actions; LLM supports coders, does not replace them
- **Synthetic data only**. No PHI; safe for sharing and evaluation

## Troubleshooting

- **Python version:** Use Python 3.12 (recommended). Python 3.13 removed the `audioop` module, which can break Gradio 4.x on some setups.
- **First run is slow:** Sentence-BERT downloads model weights (~90 MB) on first launch.
- **API errors:** Confirm `OPENAI_API_KEY` is set in `.env` or entered in the UI sidebar.

## Limitations

- Sample notes are synthetic, not production chart data
- Entity extraction is regex-based, not a clinical NER model
- PDF support depends on text-layer extraction (scanned images would need OCR - a natural next step toward multimodal)
- Requires an OpenAI API key for generation (retrieval works offline after model download)
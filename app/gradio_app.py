import os
import sys
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

load_dotenv(ROOT_DIR / ".env")

from src.clinical_notes_loader import ClinicalNotesLoader
from src.document_processor import DocumentProcessor
from src.rag_agent import DEFAULT_OLLAMA_MODEL, RAGAgent
from src.vector_store import VectorStore, get_embedding_model

EXAMPLE_QUERIES = [
    "What ICD-10 codes apply to the diabetic patient with hypertension and elevated A1C?",
    "Does the COPD patient documentation support tobacco-use coding for risk adjustment?",
    "What documentation gaps exist for the CKD patient with Type 2 diabetes?",
    "Which codes and supporting evidence apply to the atrial fibrillation case?",
]

ESSAY_CONTEXT = """
**Essay tie-in:** This prototype implements the *RAG-Powered Coder Decision Support*
recommendation from my Clinical NLP essay - surfacing relevant clinical evidence and coding
context at chart review time, with transparent retrieval and human-in-the-loop guardrails
aligned with Cotiviti's risk adjustment workflows.
"""


def _index_documents(text: str, source_label: str, vector_store: VectorStore | None = None):
    processor = DocumentProcessor()
    chunks = processor.chunk_document(text)
    entities = processor.extract_clinical_entities(text)

    if vector_store is None:
        vector_store = VectorStore()
    else:
        vector_store.clear()
    vector_store.add_documents(chunks)

    rag_agent = RAGAgent(vector_store)

    status = (
        f"Loaded **{len(chunks)} chunks** from {source_label}. "
        f"Detected **{len(entities['icd10_codes'])} ICD-10 codes**."
    )
    entity_summary = processor.format_entities(entities)
    return vector_store, rag_agent, status, entity_summary


def load_sample_data(vector_store=None):
    text = ClinicalNotesLoader().load_sample_notes()
    return _index_documents(text, "built-in sample clinical notes", vector_store)


def load_uploaded_file(file_obj, vector_store=None):
    if file_obj is None:
        return vector_store, None, "No file uploaded.", ""

    processor = DocumentProcessor()
    filename = Path(file_obj).name if isinstance(file_obj, str) else "uploaded file"

    try:
        if isinstance(file_obj, str):
            text = processor.read_document(file_obj, filename=filename)
        else:
            raw = file_obj.read()
            text = processor.read_document(raw, filename=filename)
    except Exception as exc:
        return vector_store, None, f"Failed to read file: {exc}", ""

    return _index_documents(text, filename, vector_store)


def _format_workflow_outputs(result):
    retrieved = result.get("retrieved_chunks", [])
    retrieved_md = "\n\n---\n\n".join(
        f"**Chunk {idx}**\n{chunk}" for idx, chunk in enumerate(retrieved, start=1)
    )
    if not retrieved_md:
        retrieved_md = "_No chunks retrieved._"

    similarities = result.get("retrieval_similarities", [])
    score_line = ", ".join(f"{score:.2f}" for score in similarities) or "N/A"
    actions = "\n".join(f"- {action}" for action in result.get("suggested_actions", []))

    return (
        result["response"],
        result.get("confidence", "N/A"),
        score_line,
        actions,
        retrieved_md,
    )


def query_agent(query, vector_store, rag_agent):
    empty = ("Please load data first (sample or upload).", "", "", "", "")
    if vector_store is None or rag_agent is None:
        yield empty
        return

    if not query.strip():
        yield ("Please enter a question.", "", "", "", "")
        return

    for result in rag_agent.agentic_workflow_stream(query):
        yield _format_workflow_outputs(result)


def set_ollama_model(model: str):
    if model.strip():
        os.environ["OLLAMA_MODEL"] = model.strip()


def warm_up_models():
    """Pre-load embedding model and warm Ollama before the first query."""
    get_embedding_model()
    vector_store, rag_agent, status, entities = load_sample_data()
    rag_agent.warm_up()
    return vector_store, rag_agent, status, entities


with gr.Blocks(title="Agentic RAG for Risk Adjustment") as demo:
    gr.Markdown(
        """
# Agentic RAG for Risk Adjustment Decision Support

A hackathon proof-of-concept for **clinical NLP + retrieval-augmented generation**
supporting medical coders during chart review.
"""
    )
    gr.Markdown(ESSAY_CONTEXT)

    vector_store_state = gr.State(None)
    rag_agent_state = gr.State(None)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Setup (Ollama - free, local)")
            gr.Markdown(
                "Install [Ollama](https://ollama.com), then run `ollama pull llama3.2:1b` once."
            )
            ollama_model = gr.Textbox(
                label="Ollama Model",
                placeholder="llama3.2:1b",
                value=os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
                info="Default: llama3.2:1b (fast on CPU). For better quality: llama3.2",
            )
            ollama_model.change(set_ollama_model, inputs=ollama_model)

            gr.Markdown("### Load Clinical Notes")
            load_sample_btn = gr.Button("Reload Sample Notes", variant="secondary")
            upload_file = gr.File(
                label="Upload .txt or .pdf",
                file_types=[".txt", ".pdf"],
            )
            load_upload_btn = gr.Button("Process Upload")
            load_status = gr.Markdown()

            gr.Markdown("### Extracted Entities")
            entity_output = gr.Markdown()

        with gr.Column(scale=2):
            gr.Markdown("### Ask a Coding Question")
            query_input = gr.Textbox(
                label="Question",
                placeholder="e.g., What ICD-10 codes for a diabetic patient with hypertension?",
                lines=3,
            )
            gr.Examples(
                examples=[[q] for q in EXAMPLE_QUERIES],
                inputs=query_input,
                label="Try an example",
            )
            submit_btn = gr.Button("Run Agentic Workflow", variant="primary")

            gr.Markdown("### Coder Response")
            response_output = gr.Textbox(label="Answer", lines=10, interactive=False)

            with gr.Row():
                confidence_output = gr.Textbox(label="Retrieval Confidence", interactive=False)
                retrieval_scores_output = gr.Textbox(
                    label="Similarity Scores (higher = better match)",
                    interactive=False,
                )

            actions_output = gr.Textbox(
                label="Suggested Next Actions",
                lines=4,
                interactive=False,
            )

            with gr.Accordion("Retrieved Source Chunks (audit trail)", open=False):
                retrieved_output = gr.Markdown()

    demo.load(
        warm_up_models,
        outputs=[vector_store_state, rag_agent_state, load_status, entity_output],
    )
    load_sample_btn.click(
        load_sample_data,
        inputs=[vector_store_state],
        outputs=[vector_store_state, rag_agent_state, load_status, entity_output],
    )
    load_upload_btn.click(
        load_uploaded_file,
        inputs=[upload_file, vector_store_state],
        outputs=[vector_store_state, rag_agent_state, load_status, entity_output],
    )
    submit_btn.click(
        query_agent,
        inputs=[query_input, vector_store_state, rag_agent_state],
        outputs=[
            response_output,
            confidence_output,
            retrieval_scores_output,
            actions_output,
            retrieved_output,
        ],
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(primary_hue="green"))

#!/usr/bin/env python3
"""Headless demo of the agentic RAG pipeline (no UI required)."""

import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
load_dotenv(ROOT_DIR / ".env")

from openai import OpenAI

from src.clinical_notes_loader import ClinicalNotesLoader
from src.document_processor import DocumentProcessor
from src.rag_agent import DEFAULT_OLLAMA_BASE_URL, DEFAULT_OLLAMA_MODEL, RAGAgent
from src.vector_store import VectorStore

DEMO_QUESTIONS = [
    "What ICD-10 codes apply to the diabetic patient with hypertension and elevated A1C?",
    "Does the COPD patient documentation support tobacco-use coding for risk adjustment?",
]


def check_ollama() -> bool:
    import os

    base_url = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
    model = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    try:
        client = OpenAI(base_url=base_url, api_key="ollama", timeout=10.0)
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=5,
        )
        return True
    except Exception as exc:
        print(f"Cannot reach Ollama ({base_url}, model={model}): {exc}")
        print("\nSetup:")
        print("  1. Install Ollama from https://ollama.com")
        print(f"  2. Run: ollama pull {model}")
        print("  3. Ensure Ollama is running, then retry.")
        return False


def main() -> None:
    print("=" * 72)
    print("Agentic RAG - Headless Demo")
    print("=" * 72)

    if not check_ollama():
        sys.exit(1)

    text = ClinicalNotesLoader().load_sample_notes()
    processor = DocumentProcessor()
    chunks = processor.chunk_document(text)
    entities = processor.extract_clinical_entities(text)

    print(f"\nIndexed {len(chunks)} chunks from sample clinical notes.")
    print(f"Detected ICD-10 codes: {', '.join(entities['icd10_codes']) or 'none'}")

    vector_store = VectorStore()
    vector_store.add_documents(chunks)
    agent = RAGAgent(vector_store)

    for idx, question in enumerate(DEMO_QUESTIONS, start=1):
        print("\n" + "-" * 72)
        print(f"Question {idx}: {question}")
        print("-" * 72)

        result = agent.agentic_workflow(question)
        print(f"\nConfidence: {result['confidence']}")
        print(f"Similarities: {result.get('retrieval_similarities', [])}")
        print("\nAnswer:")
        print(result["response"])
        print("\nSuggested actions:")
        for action in result.get("suggested_actions", []):
            print(f"  * {action}")

    print("\n" + "=" * 72)
    print("Demo complete. Launch the UI with: python app/gradio_app.py")
    print("=" * 72)


if __name__ == "__main__":
    main()

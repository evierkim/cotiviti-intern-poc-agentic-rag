#!/usr/bin/env python3
"""Headless demo of the agentic RAG pipeline (no UI required)."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
load_dotenv(ROOT_DIR / ".env")

from src.clinical_notes_loader import ClinicalNotesLoader
from src.document_processor import DocumentProcessor
from src.rag_agent import RAGAgent
from src.vector_store import VectorStore

DEMO_QUESTIONS = [
    "What ICD-10 codes apply to the diabetic patient with hypertension and elevated A1C?",
    "Does the COPD patient documentation support tobacco-use coding for risk adjustment?",
]


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY in .env before running the demo.")
        sys.exit(1)

    print("=" * 72)
    print("Agentic RAG - Headless Demo")
    print("=" * 72)

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

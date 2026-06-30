# Agentic RAG for Risk Adjustment Decision Support

## Overview
This Proof of Concept demonstrates an **Agentic RAG (Retrieval-Augmented Generation) System** for clinical decision support in risk adjustment and medical coding.

## Key Features
- **Document Processing**: Chunks clinical notes and extracts entities
- **Vector Search**: FAISS-based semantic retrieval of relevant documents
- **Agentic Workflow**: Retrieve -> Generate -> Self-evaluate -> Suggest next actions
- **Interactive UI**: Streamlit interface for querying clinical data
- **Confidence Scoring**: Self-evaluation of response reliability

## Technology Stack
- **Frontend**: Streamlit
- **LLM**: OpenAI GPT-3.5-Turbo (or local with Ollama)
- **Embeddings**: Sentence-BERT (all-MiniLM-L6-v2)
- **Vector Search**: FAISS
- **Language**: Python 3.12

## Quick Start

### Prerequisites
```bash
# clone the repo
git clone https://github.com/yourusername/cotiviti-intern-poc-agentic-rag.git
cd cotiviti-intern-poc-agentic-rag

# virtual environment for Windows
python -m venv venv
venv\Scripts\activate

# dependencies
pip install -r requirements.txt
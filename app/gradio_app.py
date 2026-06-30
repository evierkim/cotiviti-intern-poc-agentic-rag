import gradio as gr
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore
from src.rag_agent import RAGAgent
from src.clinical_notes_loader import ClinicalNotesLoader

# Global state variables (will be stored in session state via gr.State)
class AppState:
    def __init__(self):
        self.vector_store = None
        self.rag_agent = None
        self.documents_loaded = False

def load_sample_data():
    """Load the built‑in sample clinical notes."""
    loader = ClinicalNotesLoader()
    text = loader.load_sample_notes()
    
    processor = DocumentProcessor()
    chunks = processor.chunk_document(text)
    
    vector_store = VectorStore()
    vector_store.add_documents(chunks)
    
    rag_agent = RAGAgent(vector_store, api_key=os.getenv('OPENAI_API_KEY'))
    
    return vector_store, rag_agent, f"Loaded {len(chunks)} chunks from sample data."

def load_uploaded_file(file_obj):
    """Process an uploaded .txt or .pdf file."""
    if file_obj is None:
        return None, None, "No file uploaded."
    
    # Read content
    content = file_obj.read()
    try:
        # Try to decode as text
        text = content.decode('utf-8', errors='ignore')
    except:
        # Fallback for binary (e.g., PDF) – simple placeholder
        text = str(content)
    
    processor = DocumentProcessor()
    chunks = processor.chunk_document(text)
    
    vector_store = VectorStore()
    vector_store.add_documents(chunks)
    
    rag_agent = RAGAgent(vector_store, api_key=os.getenv('OPENAI_API_KEY'))
    
    return vector_store, rag_agent, f"Loaded {len(chunks)} chunks from uploaded file."

def query_agent(query, vector_store, rag_agent):
    """Run the agentic workflow and format output."""
    if vector_store is None or rag_agent is None:
        return "Please load data first (sample or upload).", "", "", ""
    
    if not query.strip():
        return "Please enter a question.", "", "", ""
    
    result = rag_agent.agentic_workflow(query)
    
    response_text = result['response']
    confidence = result.get('confidence', 'N/A')
    retrieval_scores = str(result.get('retrieval_scores', []))
    actions = "\n".join([f"- {a}" for a in result.get('suggested_actions', [])])
    
    return response_text, confidence, retrieval_scores, actions

# Build the Gradio interface
with gr.Blocks(title="Agentic RAG for Risk Adjustment", theme=gr.themes.Soft()) as demo:
    gr.Markdown("#Agentic RAG for Risk Adjustment Decision Support")
    
    # State variables (persist across interactions)
    vector_store_state = gr.State(None)
    rag_agent_state = gr.State(None)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Configuration")
            api_key = gr.Textbox(
                label="OpenAI API Key",
                type="password",
                placeholder="sk-...",
                info="Enter your key or set OPENAI_API_KEY in .env"
            )
            # Update environment if key provided
            def set_api_key(key):
                if key:
                    os.environ['OPENAI_API_KEY'] = key
            api_key.change(set_api_key, inputs=api_key)
            
            gr.Markdown("### Document Loading")
            load_sample_btn = gr.Button("Load Sample Data")
            upload_file = gr.File(label="Upload Clinical Notes", file_types=[".txt", ".pdf"])
            load_upload_btn = gr.Button("Process Uploaded File")
            load_status = gr.Textbox(label="Status", interactive=False)
        
        with gr.Column(scale=2):
            gr.Markdown("### Ask a Clinical Coding Question")
            query_input = gr.Textbox(
                label="Your Question",
                placeholder="e.g., What ICD-10 codes for a diabetic patient with hypertension?",
                lines=3
            )
            submit_btn = gr.Button("Generate Response", variant="primary")
            
            gr.Markdown("### Response")
            response_output = gr.Textbox(label="Response", lines=8, interactive=False)
            
            with gr.Row():
                with gr.Column():
                    confidence_output = gr.Textbox(label="Confidence", interactive=False)
                with gr.Column():
                    retrieval_scores_output = gr.Textbox(label="Retrieval Scores", interactive=False)
            
            actions_output = gr.Textbox(label="Suggested Next Actions", lines=3, interactive=False)
    
    # Event handlers
    load_sample_btn.click(
        load_sample_data,
        inputs=[],
        outputs=[vector_store_state, rag_agent_state, load_status]
    )
    
    load_upload_btn.click(
        load_uploaded_file,
        inputs=[upload_file],
        outputs=[vector_store_state, rag_agent_state, load_status]
    )
    
    submit_btn.click(
        query_agent,
        inputs=[query_input, vector_store_state, rag_agent_state],
        outputs=[response_output, confidence_output, retrieval_scores_output, actions_output]
    )

if __name__ == "__main__":
    demo.launch()
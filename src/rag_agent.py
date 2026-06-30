import os
from typing import List, Dict, Any
from openai import OpenAI
from src.vector_store import VectorStore

class RAGAgent:
    def __init__(self, vector_store: VectorStore, api_key: str = None):
        self.vector_store = vector_store
        self.client = OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
        self.conversation_history = []
    
    def retrieve_context(self, query: str, k: int = 3) -> str:
        """Retrieve relevant context from vector store."""
        results = self.vector_store.search(query, k)
        context_parts = []
        for i, result in enumerate(results):
            context_parts.append(f"[Document {i+1}]\n{result['document']['text']}\n")
        return "\n".join(context_parts)
    
    def generate_response(self, query: str, context: str) -> Dict[str, Any]:
        """Generate response using LLM with retrieved context."""
        system_prompt = """You are a clinical decision support assistant for medical coders at a healthcare analytics company.
        
        Your role is to:
        1. Retrieve relevant clinical information from provided context
        2. Help coders identify appropriate ICD-10 codes and documentation requirements
        3. Provide reasoning for your recommendations
        4. Flag any uncertainty or need for clinician review
        
        Always cite specific information from the context when making recommendations.
        Be conservative - do not hallucinate codes or diagnoses not mentioned in context."""
        
        user_prompt = f"""Context from clinical documents:
        {context}
        
        Question: {query}
        
        Please provide a reasoned response with:
        1. Relevant clinical findings from the context
        2. Suggested ICD-10 codes (if applicable)
        3. Confidence level in your recommendation
        4. Any gaps in documentation that need to be addressed"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            return {
                'response': response.choices[0].message.content,
                'context_used': context,
                'query': query
            }
        except Exception as e:
            return {
                'response': f"Error generating response: {str(e)}",
                'context_used': context,
                'query': query
            }
    
    def agentic_workflow(self, query: str) -> Dict[str, Any]:
        """Complete agentic workflow: retrieve → generate → self-evaluate."""
        # Step 1: Retrieve context
        context = self.retrieve_context(query)
        
        # Step 2: Generate response
        response_data = self.generate_response(query, context)
        
        # Step 3: Self-evaluation (simple confidence based on retrieval scores)
        search_results = self.vector_store.search(query, 3)
        avg_score = 0
        if search_results:
            avg_score = sum(r['score'] for r in search_results) / len(search_results)
        
        confidence = "High" if avg_score < 0.8 else "Medium" if avg_score < 1.5 else "Low"
        
        response_data['confidence'] = confidence
        response_data['retrieval_scores'] = [r['score'] for r in search_results]
        
        # Step 4: Suggest next actions (Agentic behavior)
        actions = []
        if "documentation" in query.lower() or "coding" in query.lower():
            actions.append("Review the original clinical note for additional details")
        if "medication" in query.lower():
            actions.append("Check medication reconciliation for completeness")
        if "diagnosis" in query.lower() or "diagnoses" in query.lower():
            actions.append("Verify all diagnoses are clearly documented and coded")
        
        response_data['suggested_actions'] = actions
        
        return response_data
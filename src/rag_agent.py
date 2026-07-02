import os
from typing import Any, Dict, Generator, List

from openai import OpenAI

from src.vector_store import VectorStore

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"
DEFAULT_OLLAMA_MODEL = "llama3.2:1b"
DEFAULT_RETRIEVAL_K = 2
DEFAULT_MAX_TOKENS = 300

SYSTEM_PROMPT = """You are a clinical coding assistant for risk adjustment.
Use only the provided context. Suggest ICD-10 codes supported by the evidence,
note documentation gaps, and flag when clinician review is needed. Be concise."""


class RAGAgent:
    def __init__(
        self,
        vector_store: VectorStore,
        base_url: str | None = None,
        model: str | None = None,
    ):
        self.vector_store = vector_store
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
        self.model = model or os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
        self.client = OpenAI(
            base_url=self.base_url,
            api_key="ollama",
            timeout=90.0,
        )

    def retrieve(self, query: str, k: int = DEFAULT_RETRIEVAL_K) -> List[Dict[str, Any]]:
        return self.vector_store.search(query, k)

    def _format_context(self, results: List[Dict[str, Any]]) -> str:
        if not results:
            return "No relevant documents retrieved."

        parts = []
        for i, result in enumerate(results, start=1):
            parts.append(f"[{i}] {result['document']['text']}")
        return "\n\n".join(parts)

    def _similarity_from_distance(self, distance: float) -> float:
        return 1.0 / (1.0 + distance)

    def _active_model(self) -> str:
        return os.getenv("OLLAMA_MODEL", self.model)

    def _llm_setup_hint(self) -> str:
        return (
            f"Using Ollama at {self.base_url} with model '{self._active_model()}'.\n\n"
            "Setup:\n"
            "  1. Install Ollama from https://ollama.com\n"
            "  2. Run: ollama pull llama3.2:1b\n"
            "  3. Ensure Ollama is running (it starts automatically after install)\n\n"
            "For better quality on GPU machines, try: ollama pull llama3.2"
        )

    def _chat_messages(self, query: str, context: str) -> List[Dict[str, str]]:
        user_prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Briefly list: (1) key findings, (2) ICD-10 codes, "
            "(3) gaps/risks, (4) clinician review needed?"
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

    def warm_up(self) -> None:
        """Load the Ollama model into memory before the first user query."""
        try:
            self.client.chat.completions.create(
                model=self._active_model(),
                messages=[{"role": "user", "content": "ok"}],
                max_tokens=1,
                temperature=0,
            )
        except Exception:
            pass

    def generate_response(self, query: str, context: str) -> Dict[str, Any]:
        try:
            response = self.client.chat.completions.create(
                model=self._active_model(),
                messages=self._chat_messages(query, context),
                temperature=0.2,
                max_tokens=DEFAULT_MAX_TOKENS,
            )
            return {
                "response": response.choices[0].message.content,
                "context_used": context,
                "query": query,
            }
        except Exception as exc:
            return {
                "response": (
                    f"Error generating response: {exc}\n\n"
                    f"{self._llm_setup_hint()}"
                ),
                "context_used": context,
                "query": query,
            }

    def generate_response_stream(
        self, query: str, context: str
    ) -> Generator[str, None, None]:
        try:
            stream = self.client.chat.completions.create(
                model=self._active_model(),
                messages=self._chat_messages(query, context),
                temperature=0.2,
                max_tokens=DEFAULT_MAX_TOKENS,
                stream=True,
            )
            full = ""
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                full += delta
                yield full
        except Exception as exc:
            yield (
                f"Error generating response: {exc}\n\n"
                f"{self._llm_setup_hint()}"
            )

    def _confidence_label(self, results: List[Dict[str, Any]]) -> str:
        if not results:
            return "Low - no relevant context retrieved"

        similarities = [
            self._similarity_from_distance(result["score"]) for result in results
        ]
        avg_similarity = sum(similarities) / len(similarities)

        if avg_similarity >= 0.55:
            return f"High ({avg_similarity:.0%} avg retrieval match)"
        if avg_similarity >= 0.40:
            return f"Medium ({avg_similarity:.0%} avg retrieval match)"
        return f"Low ({avg_similarity:.0%} avg retrieval match)"

    def _suggest_actions(
        self, query: str, results: List[Dict[str, Any]], response_text: str
    ) -> List[str]:
        actions: List[str] = []
        query_lower = query.lower()
        combined_context = " ".join(
            result["document"]["text"] for result in results
        ).lower()
        response_lower = response_text.lower()

        if not results:
            actions.append("Load clinical notes or upload a chart before querying.")
            return actions

        if any(
            term in query_lower
            for term in ("icd", "code", "coding", "risk adjustment", "hcc")
        ):
            actions.append(
                "Cross-check suggested ICD-10 codes against the original source note."
            )

        if "medication" in query_lower or "drug" in query_lower:
            actions.append("Validate medication list against pharmacy and claims data.")

        if "documentation" in query_lower or "gap" in query_lower:
            actions.append(
                "Flag documentation gaps for physician query or pre-visit prep follow-up."
            )

        if "tobacco" in query_lower or "smok" in query_lower:
            actions.append(
                "Confirm tobacco-use codes (e.g., Z72.0) are explicitly documented."
            )

        if "clinician" in response_lower or "review" in response_lower:
            actions.append("Route case for clinician review before final coding submission.")

        if "e11" in combined_context and "i10" in combined_context:
            actions.append(
                "Evaluate comorbidity capture for diabetes with hypertension (risk adjustment)."
            )

        if not actions:
            actions.append("Review retrieved source chunks to confirm coding rationale.")

        actions.append(
            "Maintain human-in-the-loop oversight - LLM output supports, not replaces, coders."
        )
        return actions

    def _workflow_metadata(
        self, query: str, results: List[Dict[str, Any]], response_text: str
    ) -> Dict[str, Any]:
        return {
            "confidence": self._confidence_label(results),
            "retrieval_scores": [round(result["score"], 4) for result in results],
            "retrieval_similarities": [
                round(self._similarity_from_distance(result["score"]), 4)
                for result in results
            ],
            "retrieved_chunks": [result["document"]["text"] for result in results],
            "suggested_actions": self._suggest_actions(query, results, response_text),
        }

    def agentic_workflow(self, query: str) -> Dict[str, Any]:
        """Retrieve -> generate -> self-evaluate -> suggest next actions."""
        results = self.retrieve(query)
        context = self._format_context(results)
        response_data = self.generate_response(query, context)
        response_data.update(
            self._workflow_metadata(query, results, response_data["response"])
        )
        return response_data

    def agentic_workflow_stream(
        self, query: str
    ) -> Generator[Dict[str, Any], None, None]:
        """Streaming variant for Gradio: yields partial then final workflow state."""
        results = self.retrieve(query)
        context = self._format_context(results)
        metadata = self._workflow_metadata(query, results, "")

        yield {
            "response": "_Generating answer..._",
            "context_used": context,
            "query": query,
            **metadata,
        }

        full_response = ""
        for partial in self.generate_response_stream(query, context):
            full_response = partial
            yield {
                "response": full_response,
                "context_used": context,
                "query": query,
                **metadata,
            }

        metadata = self._workflow_metadata(query, results, full_response)
        yield {
            "response": full_response,
            "context_used": context,
            "query": query,
            **metadata,
        }

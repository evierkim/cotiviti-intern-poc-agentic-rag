import os
from typing import Any, Dict, List

from openai import OpenAI

from src.vector_store import VectorStore


class RAGAgent:
    def __init__(self, vector_store: VectorStore, api_key: str | None = None):
        self.vector_store = vector_store
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def retrieve(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        return self.vector_store.search(query, k)

    def _format_context(self, results: List[Dict[str, Any]]) -> str:
        if not results:
            return "No relevant documents retrieved."

        parts = []
        for i, result in enumerate(results, start=1):
            similarity = self._similarity_from_distance(result["score"])
            parts.append(
                f"[Chunk {i} | similarity={similarity:.2f}]\n"
                f"{result['document']['text']}\n"
            )
        return "\n".join(parts)

    def _similarity_from_distance(self, distance: float) -> float:
        return 1.0 / (1.0 + distance)

    def generate_response(self, query: str, context: str) -> Dict[str, Any]:
        system_prompt = """You are a clinical decision support assistant for medical coders
at a healthcare analytics company focused on risk adjustment.

Your role is to:
1. Use only the provided clinical context
2. Help coders identify appropriate ICD-10 codes and documentation requirements
3. Explain your reasoning clearly for auditability
4. Flag uncertainty and recommend clinician review when evidence is incomplete

Always cite specific findings from the context.
Be conservative - do not invent diagnoses, codes, or medications not supported by the context."""

        user_prompt = f"""Context from retrieved clinical documents:
{context}

Question: {query}

Respond with:
1. Relevant clinical findings from the context
2. Suggested ICD-10 codes (if supported by the context)
3. Documentation gaps or coding risks
4. Whether clinician review is recommended"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=600,
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
                    "Verify your OpenAI API key in .env or the sidebar field."
                ),
                "context_used": context,
                "query": query,
            }

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

    def agentic_workflow(self, query: str) -> Dict[str, Any]:
        """Retrieve -> generate -> self-evaluate -> suggest next actions."""
        results = self.retrieve(query, k=3)
        context = self._format_context(results)
        response_data = self.generate_response(query, context)

        response_data["confidence"] = self._confidence_label(results)
        response_data["retrieval_scores"] = [
            round(result["score"], 4) for result in results
        ]
        response_data["retrieval_similarities"] = [
            round(self._similarity_from_distance(result["score"]), 4)
            for result in results
        ]
        response_data["retrieved_chunks"] = [
            result["document"]["text"] for result in results
        ]
        response_data["suggested_actions"] = self._suggest_actions(
            query, results, response_data["response"]
        )

        return response_data

import hashlib
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Union

from pypdf import PdfReader


class DocumentProcessor:
    def read_document(self, source: Union[str, bytes], filename: str = "") -> str:
        """Read plain text or PDF content from a file path or raw bytes."""
        if isinstance(source, str):
            path = Path(source)
            if path.suffix.lower() == ".pdf":
                return self._read_pdf(path.read_bytes())
            return path.read_text(encoding="utf-8", errors="ignore")

        if filename.lower().endswith(".pdf"):
            return self._read_pdf(source)

        return source.decode("utf-8", errors="ignore")

    def _read_pdf(self, content: bytes) -> str:
        reader = PdfReader(BytesIO(content))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(pages).strip()
        if not text:
            raise ValueError("No extractable text found in PDF.")
        return text

    def chunk_document(
        self, text: str, chunk_size: int = 500, overlap: int = 50
    ) -> List[Dict[str, Any]]:
        """Split document into overlapping chunks for better retrieval."""
        if "=== CLINICAL NOTE" in text:
            note_sections = re.split(r"(?=^=== CLINICAL NOTE)", text, flags=re.MULTILINE)
            chunks: List[Dict[str, Any]] = []
            for section in note_sections:
                section = section.strip()
                if section:
                    chunks.extend(self._chunk_text(section, chunk_size, overlap))
            return chunks or self._chunk_text(text, chunk_size, overlap)

        return self._chunk_text(text, chunk_size, overlap)

    def _chunk_text(
        self, text: str, chunk_size: int = 500, overlap: int = 50
    ) -> List[Dict[str, Any]]:
        chunks = []
        sentences = re.split(r"(?<=[.!?])\s+", text)

        current_chunk: List[str] = []
        current_size = 0

        for sentence in sentences:
            sentence_len = len(sentence.split())
            if current_size + sentence_len > chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(self._make_chunk(chunk_text))
                overlap_words = (
                    " ".join(current_chunk[-overlap:]) if overlap > 0 else ""
                )
                current_chunk = [overlap_words] if overlap_words else []
                current_size = len(overlap_words.split()) if overlap_words else 0

            current_chunk.append(sentence)
            current_size += sentence_len

        if current_chunk:
            chunks.append(self._make_chunk(" ".join(current_chunk)))

        return chunks

    def _make_chunk(self, chunk_text: str) -> Dict[str, Any]:
        return {
            "text": chunk_text,
            "chunk_id": hashlib.md5(chunk_text.encode()).hexdigest()[:8],
            "metadata": {"chunk_size": len(chunk_text.split())},
        }

    def extract_clinical_entities(self, text: str) -> Dict[str, List[Any]]:
        """Regex-based clinical entity extraction for demo transparency."""
        icd_pattern = r"\b[A-TV-Z][0-9][0-9AB](?:\.[0-9A-Z]{1,4})?\b"
        icd_codes = sorted(set(re.findall(icd_pattern, text)))

        med_patterns = [
            "Metformin",
            "Lisinopril",
            "Tiotropium",
            "Albuterol",
            "Sertraline",
            "Losartan",
            "Glipizide",
            "Warfarin",
            "Metoprolol",
        ]
        medications = [med for med in med_patterns if med in text]

        lab_pattern = r"([A-Za-z0-9\-]+):\s*([0-9.]+%?)"
        lab_values = re.findall(lab_pattern, text)

        diagnosis_keywords = [
            "Diabetes",
            "Hypertension",
            "COPD",
            "Atrial Fibrillation",
            "Chronic Kidney Disease",
            "postpartum depression",
        ]
        diagnoses = [term for term in diagnosis_keywords if term.lower() in text.lower()]

        return {
            "diagnoses": diagnoses,
            "medications": medications,
            "lab_values": lab_values,
            "icd10_codes": icd_codes,
        }

    def format_entities(self, entities: Dict[str, List[Any]]) -> str:
        """Format extracted entities for display in the UI."""
        lines = ["**Extracted Clinical Entities**", ""]
        for label, key in [
            ("Diagnoses", "diagnoses"),
            ("Medications", "medications"),
            ("ICD-10 Codes", "icd10_codes"),
        ]:
            values = entities.get(key, [])
            display = ", ".join(values) if values else "None detected"
            lines.append(f"- **{label}:** {display}")

        labs = entities.get("lab_values", [])
        if labs:
            lab_display = ", ".join(f"{name}: {value}" for name, value in labs[:8])
            lines.append(f"- **Lab Values:** {lab_display}")
        else:
            lines.append("- **Lab Values:** None detected")

        return "\n".join(lines)

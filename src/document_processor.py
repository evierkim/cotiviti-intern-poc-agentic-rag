import re
import hashlib
from typing import List, Dict, Any

class DocumentProcessor:
    def __init__(self):
        pass
    
    def chunk_document(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict[str, Any]]:
        """Split document into overlapping chunks for better retrieval."""
        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_len = len(sentence.split())
            if current_size + sentence_len > chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'chunk_id': hashlib.md5(chunk_text.encode()).hexdigest()[:8],
                    'metadata': {'chunk_size': len(chunk_text.split())}
                })
                # Keep overlap
                overlap_words = ' '.join(current_chunk[-overlap:]) if overlap > 0 else ''
                current_chunk = [overlap_words] if overlap_words else []
                current_size = len(overlap_words.split()) if overlap_words else 0
            
            current_chunk.append(sentence)
            current_size += sentence_len
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'chunk_id': hashlib.md5(chunk_text.encode()).hexdigest()[:8],
                'metadata': {'chunk_size': len(chunk_text.split())}
            })
        
        return chunks
    
    def extract_clinical_entities(self, text: str) -> Dict[str, List[str]]:
        """Simple regex-based entity extraction (demonstration)."""
        entities = {
            'diagnoses': [],
            'medications': [],
            'lab_values': [],
            'coding': []
        }
        
        # ICD-10 code pattern
        icd_pattern = r'[A-Z][0-9][0-9]\.?[0-9]*'
        entities['coding'] = re.findall(icd_pattern, text)
        
        # Common medications (simplified)
        med_patterns = ['Metformin', 'Lisinopril', 'Tiotropium', 'Albuterol', 'Sertraline', 'Losartan', 'Glipizide', 'Warfarin', 'Metoprolol']
        for med in med_patterns:
            if med in text:
                entities['medications'].append(med)
        
        # Lab values (e.g., A1C: 8.2%)
        lab_pattern = r'([A-Za-z0-9]+):\s*([0-9.]+)'
        entities['lab_values'] = re.findall(lab_pattern, text)
        
        return entities
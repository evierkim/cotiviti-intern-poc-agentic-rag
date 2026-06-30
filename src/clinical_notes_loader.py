import os
from pathlib import Path

class ClinicalNotesLoader:
    def load_sample_notes(self) -> str:
        """Load sample clinical notes from the sample_notes.txt file."""
        sample_path = Path(__file__).parent.parent / "sample_notes.txt"
        if sample_path.exists():
            with open(sample_path, 'r') as f:
                return f.read()
        else:
            # Fallback sample data
            return self._get_fallback_data()
    
    def _get_fallback_data(self) -> str:
        return """=== CLINICAL NOTE 1 ===
Patient: 45-year-old female with Type 2 Diabetes (E11.9)
A1C: 8.2% (elevated)
BMI: 34.5
Hypertension present (I10)
Current medications: Metformin 1000mg BID, Lisinopril 10mg daily
Recommendation: Increase Metformin to 1500mg BID, consider GLP-1 agonist
Coding notes: E11.9, I10, Z79.84 (long-term metformin use)

=== CLINICAL NOTE 2 ===
Patient: 67-year-old male with COPD (J44.9)
Smoking history: 40 pack-years, current smoker
Oxygen saturation: 89% on room air
Tiotropium and albuterol PRN
Recommendation: Initiate home oxygen therapy, smoking cessation counseling
Coding notes: J44.9, Z72.0 (tobacco use), Z87.891 (personal history of tobacco dependence)"""
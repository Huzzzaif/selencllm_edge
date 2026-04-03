from __future__ import annotations

"""
agents/agent1_detector.py
Agent 1 — PHI Detector

Single responsibility: detect PHI spans in a sentence using the LLM.
Returns list of {span, phi_type, confidence} — nothing else.

Parameters controlled by Agent 3:
  sensitivity       float 0.3–1.0  — detection threshold (higher = more aggressive)
  phi_types_focus   list[str]      — PHI types to emphasise after detected misses
"""

import json
import re
import ollama
from config import LLM_MODEL


DETECT_PROMPT = """You are a PHI (Protected Health Information) detection engine running on an edge device.

Given a sentence, identify sensitive spans that could identify a person.
Detection sensitivity: {sensitivity:.1f} (0.3=conservative, flag only obvious PHI; 1.0=aggressive, flag ambiguous references too)
{focus_instruction}
PHI categories:
- PERSON: names of people
- LOCATION: addresses, cities, states, zip codes
- DATE: birthdates, appointment dates, specific dates tied to a person
- AGE: specific ages
- DIAGNOSIS: medical conditions, diseases, medications
- ORG: hospital names, clinic names, insurance companies
- SSN: social security numbers
- PHONE: phone numbers
- EMAIL: email addresses
- ID_NUMBER: patient IDs, license numbers, account numbers
- CREDIT_CARD: credit card numbers
- RELATIONSHIP: spouse, son, daughter, wife, husband references

CRITICAL SPAN RULES:
- Span must be the MINIMUM sensitive token only — the actual value, not surrounding label words.
- Do NOT include context words like "SSN", "Patient", "Name:", "DOB:", etc. adjacent to the value.
- Correct: "Patient SSN is 123-45-6789" → span is "123-45-6789"
- Correct: "Dr. Ahmed prescribed" → span is "Dr. Ahmed"

Respond ONLY with a JSON array. No explanation. No markdown. No preamble.
Each entry: {{"span": "exact text from sentence", "phi_type": "CATEGORY", "confidence": 0.0-1.0}}
If no PHI found, return: []

Sentence: {sentence}
"""


class Agent1Detector:
    def __init__(self, params: dict):
        self.model           = LLM_MODEL
        self.sensitivity     = float(params.get("sensitivity", 0.7))
        self.phi_types_focus = list(params.get("phi_types_focus", []))

    def update_params(self, params: dict):
        self.sensitivity     = float(params.get("sensitivity", self.sensitivity))
        self.phi_types_focus = list(params.get("phi_types_focus", self.phi_types_focus))

    def detect(self, sentence: str) -> list[dict]:
        """
        Detect PHI spans in sentence.
        Returns list of {span, phi_type, confidence}.
        Only spans with confidence >= (1 - sensitivity) are returned.
        """
        focus_instruction = ""
        if self.phi_types_focus:
            types_str = ", ".join(self.phi_types_focus)
            focus_instruction = (
                f"PRIORITY: Pay extra attention to these categories "
                f"that have recently been missed: {types_str}\n"
            )

        prompt = DETECT_PROMPT.format(
            sentence=sentence,
            sensitivity=self.sensitivity,
            focus_instruction=focus_instruction,
        )
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0},
            )
            raw = response["message"]["content"].strip()
            raw = re.sub(r"```json|```", "", raw).strip()
            raw = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", raw)

            spans = json.loads(raw)
            min_conf = 1.0 - self.sensitivity
            valid = []
            for s in spans:
                if not (isinstance(s, dict) and "span" in s and "phi_type" in s):
                    continue
                if s["span"] not in sentence:
                    continue
                confidence = float(s.get("confidence", 0.5))
                if confidence < min_conf:
                    continue
                valid.append({
                    "span":        s["span"],
                    "phi_type":    s["phi_type"].upper(),
                    "confidence":  confidence,
                    "has_pattern": bool(s.get("has_pattern", False)),
                })
            return valid

        except Exception as e:
            print(f"[Agent1] Detect error: {e}")
            return []

from __future__ import annotations

"""
agents/agent_a.py
Agent A — PHI Detector, Regex Generator, Router, Masker

Pipeline per sentence:
1. Check pattern_memory for cache hits (promoted patterns first)
2. For cache misses → ask LLM to detect PHI spans + generate regex
3. Route each span: ENCRYPT or ABSTRACT
4. Return masked sentence + metadata
"""

import json
import re
import time
import ollama
from config import (
    LLM_MODEL, ENCRYPT_TYPES, ABSTRACT_TYPES,
    DEFAULT_ABSTRACTION_LEVEL
)
from memory.pattern_memory import PatternMemory
from encryption.chacha import ChaChaEncryptor


# ── Prompts ───────────────────────────────────────────────────────────────────

DETECT_PROMPT = """You are a PHI (Protected Health Information) detection engine running on an edge device.

Given a sentence, identify ALL sensitive spans that could identify a person.

PHI categories to detect:
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
- The span must be the MINIMUM sensitive token only — the actual value, not surrounding label words.
- Do NOT include context words like "SSN", "Patient", "Name:", "DOB", etc. that appear next to the value.
- Example: for "Patient SSN is 123-45-6789", the span is "123-45-6789", NOT "Patient SSN is 123-45-6789".
- Example: for "Dr. Ahmed", the span is "Dr. Ahmed" (the full name token), NOT "Dr. Ahmed prescribed".

Respond ONLY with a JSON array. No explanation. No markdown. No preamble.
Format:
[
  {{"span": "exact text from sentence", "phi_type": "CATEGORY", "has_pattern": true/false}},
  ...
]

If no PHI found, return: []

Sentence: {sentence}
"""

REGEX_PROMPT = """You are a regex generation engine.

Generate a Python regex pattern that would match this type of PHI span.
The pattern must be general enough to match similar spans, not just this exact one.

PHI type: {phi_type}
Example span: {span}

Rules:
- Return ONLY the raw regex string, nothing else
- No explanation, no markdown, no quotes
- Must be valid Python re module syntax
- Must be general (match the pattern, not just this literal string)

Example output for SSN: \\b\\d{{3}}-\\d{{2}}-\\d{{4}}\\b
"""

ABSTRACT_PROMPT = """You are a medical records anonymization system. Your task is to generalize specific identifying terms into their generic category for HIPAA compliance.

Input sentence: {sentence}
Term to generalize: {span}
Term category: {phi_type}

Output the generic category term only, using these mappings:
- PERSON → the patient, the physician, the caregiver, the specialist (pick most contextually appropriate)
- LOCATION → a residential address, an urban area, a rural area (pick most contextually appropriate)
- DIAGNOSIS → a medical condition, a chronic condition, an infectious disease (pick most contextually appropriate)
- AGE → a middle-aged adult, an elderly patient, a young adult (pick most contextually appropriate)
- ORG → a medical facility, a regional hospital, a healthcare provider (pick most contextually appropriate)
- RELATIONSHIP → a family member, a close contact, a household member (pick most contextually appropriate)
- DATE → a recent date, a specific date, a past date (pick most contextually appropriate)

Respond with ONLY the replacement phrase. 2-4 words maximum. Nothing else."""


class AgentA:
    def __init__(self, memory: PatternMemory, encryptor: ChaChaEncryptor):
        self.memory    = memory
        self.encryptor = encryptor
        self.model     = LLM_MODEL

    # ── Main Entry ────────────────────────────────────────────────────────────

    def process(self, sentence: str) -> dict:
        """
        Full pipeline for one sentence.
        Returns:
        {
            "original":        str,
            "masked":          str,
            "spans":           list of span dicts,
            "vault_snapshot":  dict,  (encrypted spans only)
            "latency_ms":      float,
            "cache_hits":      int,
            "cache_misses":    int,
        }
        """
        t_start = time.time()

        # Step 1 — regex cache scan
        cache_matches = self.memory.match(sentence)
        cache_hit_spans = {m["span"] for m in cache_matches}

        # Step 2 — LLM detection for spans not caught by cache
        llm_spans = []
        cache_misses = 0
        if not self._fully_covered(sentence, cache_matches):
            llm_spans    = self._llm_detect(sentence)
            cache_misses = len([s for s in llm_spans if s["span"] not in cache_hit_spans])

            # For new spans with patterns → generate regex + add to memory
            for span_info in llm_spans:
                if span_info["span"] not in cache_hit_spans and span_info["has_pattern"]:
                    self._generate_and_store_regex(span_info)

        # Step 3 — merge all detected spans (deduplicate)
        all_spans = self._merge_spans(cache_matches, llm_spans)

        # Step 4 — route + mask
        masked, vault_snapshot, processed_spans = self._mask(sentence, all_spans)

        latency_ms = (time.time() - t_start) * 1000

        return {
            "original":       sentence,
            "masked":         masked,
            "spans":          processed_spans,
            "vault_snapshot": vault_snapshot,
            "latency_ms":     round(latency_ms, 2),
            "cache_hits":     len(cache_matches),
            "cache_misses":   cache_misses,
        }

    # ── LLM Calls ─────────────────────────────────────────────────────────────

    def _llm_detect(self, sentence: str) -> list[dict]:
        """Ask LLM to detect PHI spans. Returns list of {span, phi_type, has_pattern}."""
        prompt = DETECT_PROMPT.format(sentence=sentence)
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0}
            )
            raw = response["message"]["content"].strip()

            # Strip markdown fences if model adds them
            raw = re.sub(r"```json|```", "", raw).strip()
            # Remove control characters that break JSON parsing (tabs/newlines outside strings)
            raw = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", raw)

            spans = json.loads(raw)
            # Validate each entry has required keys
            valid = []
            for s in spans:
                if isinstance(s, dict) and "span" in s and "phi_type" in s:
                    s.setdefault("has_pattern", False)
                    # Only include spans actually present in the sentence
                    if s["span"] in sentence:
                        valid.append(s)
            return valid

        except (json.JSONDecodeError, KeyError, Exception) as e:
            print(f"[AgentA] LLM detect error: {e}")
            return []

    def _llm_generate_regex(self, phi_type: str, span: str) -> str | None:
        """Ask LLM to generate a regex for a given PHI type + example span."""
        prompt = REGEX_PROMPT.format(phi_type=phi_type, span=span)
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0}
            )
            raw = response["message"]["content"].strip()
            raw = re.sub(r"```.*?```", "", raw, flags=re.DOTALL).strip()

            # Validate the regex compiles
            re.compile(raw)
            return raw

        except re.error:
            print(f"[AgentA] Invalid regex generated for {phi_type}: {raw}")
            return None
        except Exception as e:
            print(f"[AgentA] LLM regex error: {e}")
            return None

    def _llm_abstract(self, sentence: str, span: str, phi_type: str) -> str:
        """Ask LLM for semantic equivalent of a span in context."""
        prompt = ABSTRACT_PROMPT.format(sentence=sentence, span=span, phi_type=phi_type)
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0}
            )
            replacement = response["message"]["content"].strip()
            # Sanity: replacement shouldn't be longer than 6 words
            if len(replacement.split()) > 6:
                replacement = " ".join(replacement.split()[:6])
            return replacement

        except Exception as e:
            print(f"[AgentA] LLM abstract error: {e}")
            return "[REDACTED]"  # fallback

    # ── Routing ───────────────────────────────────────────────────────────────

    def _route(self, phi_type: str) -> str:
        """
        Returns 'encrypt' or 'abstract'.
        ENCRYPT_TYPES  → need reversibility (SSN, DOB, phone, etc.)
        ABSTRACT_TYPES → semantic replacement
        """
        if phi_type.upper() in ENCRYPT_TYPES:
            return "encrypt"
        return "abstract"

    # ── Masking ───────────────────────────────────────────────────────────────

    def _mask(self, sentence: str, spans: list[dict]) -> tuple[str, dict, list]:
        """
        Apply encrypt or abstract to each span.
        Returns masked sentence, vault snapshot, processed span list.
        """
        vault_snapshot = {}
        processed      = []

        # Sort spans by first occurrence in the original sentence (rightmost first).
        # Then re-find each span in the live `masked` string before splicing —
        # this handles offset shifts from prior replacements correctly.
        positioned = []
        for s in spans:
            idx = sentence.find(s["span"])
            if idx >= 0:
                positioned.append((idx, s))
        positioned.sort(key=lambda x: x[0], reverse=True)

        masked = sentence
        for _, span_info in positioned:
            span     = span_info["span"]
            phi_type = span_info.get("phi_type", "UNKNOWN")
            action   = self._route(phi_type)

            if action == "encrypt":
                token, record = self.encryptor.encrypt_span(span)
                token_id      = token.replace("<<ENC:", "").replace(":ENC>>", "")
                vault_snapshot[token_id] = record
                replacement   = token

            else:  # abstract
                cached_role = span_info.get("abstraction_role")
                if cached_role:
                    replacement = cached_role
                else:
                    replacement = self._llm_abstract(sentence, span, phi_type)
                    label = span_info.get("label")
                    if label:
                        self.memory.update_abstraction(label, replacement)

            # Re-find in the current masked string, then splice — never use str.replace()
            start = masked.find(span)
            if start == -1:
                continue
            masked = masked[:start] + replacement + masked[start + len(span):]

            processed.append({
                **span_info,
                "action":      action,
                "replacement": replacement,
            })

        return masked, vault_snapshot, processed

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _generate_and_store_regex(self, span_info: dict):
        """Generate regex for a new span and add to pattern memory."""
        phi_type = span_info["phi_type"]
        span     = span_info["span"]
        label    = f"{phi_type}_{span[:8].replace(' ', '_')}"

        regex = self._llm_generate_regex(phi_type, span)
        if regex:
            self.memory.add(
                label=label,
                regex=regex,
                phi_type=phi_type,
                abstraction_role=None  # filled on first abstract call
            )

    def _merge_spans(self, cache_matches: list[dict], llm_spans: list[dict]) -> list[dict]:
        """Merge cache hits and LLM detections, deduplicate by span text."""
        seen   = set()
        merged = []

        for m in cache_matches:
            if m["span"] not in seen:
                seen.add(m["span"])
                merged.append(m)

        for s in llm_spans:
            if s["span"] not in seen:
                seen.add(s["span"])
                merged.append(s)

        return merged

    def _fully_covered(self, sentence: str, cache_matches: list[dict]) -> bool:
        """
        Heuristic: if we have >= 2 promoted cache hits, skip LLM detection.
        Tunable — conservative for now.
        """
        promoted_hits = sum(1 for m in cache_matches if m.get("promoted"))
        return promoted_hits >= 2
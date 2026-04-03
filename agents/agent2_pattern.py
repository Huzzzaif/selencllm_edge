from __future__ import annotations

"""
agents/agent2_pattern.py
Agent 2 — Pattern Learner + Masker

Single responsibility: receive detected spans from Agent 1, apply or learn regex
patterns from pattern_memory, route each span to encrypt or abstract, and return
the masked sentence with full metadata.

Parameters controlled by Agent 3:
  min_confidence_to_apply  float 0.5–0.95 — minimum pattern confidence to auto-apply
  promote_threshold        int   5–50      — hits before a pattern is auto-promoted
  abstraction_level        int   1–3       — 1=role, 2=category, 3=abstract
"""

import re
import time
import ollama
from config import LLM_MODEL, ENCRYPT_TYPES
from memory.pattern_memory import PatternMemory
from encryption.chacha import ChaChaEncryptor


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

Respond with ONLY the replacement phrase. 2-4 words maximum. All lowercase. Nothing else."""


class Agent2PatternMasker:
    def __init__(self, memory: PatternMemory, encryptor: ChaChaEncryptor, params: dict):
        self.memory    = memory
        self.encryptor = encryptor
        self.model     = LLM_MODEL
        self.min_confidence_to_apply = float(params.get("min_confidence_to_apply", 0.75))
        self.promote_threshold       = int(params.get("promote_threshold", 20))
        self.abstraction_level       = int(params.get("abstraction_level", 1))

    def update_params(self, params: dict):
        self.min_confidence_to_apply = float(params.get("min_confidence_to_apply", self.min_confidence_to_apply))
        self.promote_threshold       = int(params.get("promote_threshold", self.promote_threshold))
        self.abstraction_level       = int(params.get("abstraction_level", self.abstraction_level))

    def process(self, sentence: str, llm_spans: list[dict]) -> dict:
        """
        Apply pattern memory + masking to a sentence given Agent 1's detected spans.
        Returns masked sentence, vault snapshot, and span metadata.
        """
        t_start = time.time()

        # Step 1 — regex cache scan (filter by min_confidence_to_apply)
        cache_matches = [
            m for m in self.memory.match(sentence)
            if m["confidence"] >= self.min_confidence_to_apply
        ]
        cache_hit_spans = {m["span"] for m in cache_matches}

        # Step 2 — for LLM spans not in cache, optionally learn regex
        cache_misses = 0
        for span_info in llm_spans:
            if span_info["span"] not in cache_hit_spans:
                cache_misses += 1
                if span_info.get("has_pattern", False):
                    self._generate_and_store_regex(span_info)

        # Step 3 — merge and deduplicate
        all_spans = self._merge_spans(cache_matches, llm_spans)

        # Step 4 — route + mask
        masked, vault_snapshot, processed_spans = self._mask(sentence, all_spans)

        # Step 5 — prune low-confidence patterns
        self.memory.prune()

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

    # ── Masking ───────────────────────────────────────────────────────────────

    def _mask(self, sentence: str, spans: list[dict]) -> tuple[str, dict, list]:
        vault_snapshot = {}
        processed      = []

        # Sort by first occurrence in original sentence (rightmost first), then
        # re-find in live `masked` before each splice to handle offset shifts.
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
            else:
                cached_role = span_info.get("abstraction_role")
                if cached_role:
                    replacement = cached_role
                else:
                    replacement = self._llm_abstract(sentence, span, phi_type)
                    label = span_info.get("label")
                    if label:
                        self.memory.update_abstraction(label, replacement)

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

    def _route(self, phi_type: str) -> str:
        return "encrypt" if phi_type.upper() in ENCRYPT_TYPES else "abstract"

    # ── LLM Helpers ──────────────────────────────────────────────────────────

    def _llm_abstract(self, sentence: str, span: str, phi_type: str) -> str:
        prompt = ABSTRACT_PROMPT.format(sentence=sentence, span=span, phi_type=phi_type)
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0},
            )
            replacement = response["message"]["content"].strip()
            if len(replacement.split()) > 6:
                replacement = " ".join(replacement.split()[:6])
            return replacement
        except Exception as e:
            print(f"[Agent2] Abstract error: {e}")
            return "[REDACTED]"

    def _llm_generate_regex(self, phi_type: str, span: str) -> str | None:
        prompt = REGEX_PROMPT.format(phi_type=phi_type, span=span)
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0},
            )
            raw = response["message"]["content"].strip()
            raw = re.sub(r"```.*?```", "", raw, flags=re.DOTALL).strip()
            re.compile(raw)
            return raw
        except re.error:
            print(f"[Agent2] Invalid regex for {phi_type}: {span}")
            return None
        except Exception as e:
            print(f"[Agent2] Regex error: {e}")
            return None

    def _generate_and_store_regex(self, span_info: dict):
        phi_type = span_info["phi_type"]
        span     = span_info["span"]
        label    = f"{phi_type}_{span[:8].replace(' ', '_')}"
        regex    = self._llm_generate_regex(phi_type, span)
        if regex:
            self.memory.add(label=label, regex=regex, phi_type=phi_type)

    # ── Span Utilities ────────────────────────────────────────────────────────

    def _merge_spans(self, cache_matches: list[dict], llm_spans: list[dict]) -> list[dict]:
        seen, merged = set(), []
        for m in cache_matches:
            if m["span"] not in seen:
                seen.add(m["span"])
                merged.append(m)
        for s in llm_spans:
            if s["span"] not in seen:
                seen.add(s["span"])
                merged.append(s)
        return merged

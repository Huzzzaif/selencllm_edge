"""
agents/agent_b.py
Agent B — Parallel Validator + RL Feedback

Runs concurrently with Agent A output validation:
1. Scan masked sentence for PHI leakage (missed spans)
2. Check for over-encryption (benign text encrypted)
3. Check semantic fluency of abstractions
4. Emit reward signals → update pattern_memory
5. Return validation report
"""

import json
import re
import time
import ollama
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import (
    LLM_MODEL,
    REWARD_CORRECT_REDACTION,
    REWARD_MISSED_PHI,
    REWARD_OVER_ENCRYPTION,
    REWARD_CACHE_HIT,
    REWARD_UTILITY_PRESERVED,
    REWARD_REIDENTIFIABLE,
)
from memory.pattern_memory import PatternMemory


# ── Prompts ───────────────────────────────────────────────────────────────────

VALIDATE_PROMPT = """You are a privacy auditor. Your job is to check if a masked sentence still contains any sensitive PHI (Protected Health Information) that was NOT properly masked.

Original sentence: {original}
Masked sentence: {masked}

Check the masked sentence ONLY. Identify any remaining sensitive spans that:
- Are still identifiable as a real person, place, or sensitive identifier
- Were NOT replaced or encrypted

Respond ONLY with a JSON object. No explanation. No markdown.
Format:
{{
  "leaked_spans": ["span1", "span2"],
  "is_clean": true/false,
  "confidence": 0.0-1.0
}}

If no PHI leaked, return:
{{"leaked_spans": [], "is_clean": true, "confidence": 1.0}}
"""

FLUENCY_PROMPT = """You are a linguistic quality auditor.

Check if this masked sentence is semantically fluent and coherent after PHI replacement.

Masked sentence: {masked}

Respond ONLY with a JSON object. No explanation. No markdown.
Format:
{{
  "is_fluent": true/false,
  "fluency_score": 0.0-1.0,
  "reason": "one short sentence"
}}
"""

OVERENCRYPTION_PROMPT = """You are a privacy auditor checking for over-encryption.

Original sentence: {original}
Masked sentence: {masked}
Encrypted spans (these were encrypted): {encrypted_spans}

Check if any of the encrypted spans were NOT actually sensitive PHI — i.e., benign text was encrypted unnecessarily.

Respond ONLY with a JSON object. No explanation. No markdown.
Format:
{{
  "over_encrypted": ["span1", "span2"],
  "is_accurate": true/false
}}

If all encryptions were correct, return:
{{"over_encrypted": [], "is_accurate": true}}
"""


class AgentB:
    def __init__(self, memory: PatternMemory):
        self.memory = memory
        self.model  = LLM_MODEL

    # ── Main Entry ────────────────────────────────────────────────────────────

    def validate(self, agent_a_result: dict) -> dict:
        """
        Validate Agent A output in parallel checks.
        Returns full validation report + reward signals applied.

        agent_a_result keys:
            original, masked, spans, vault_snapshot,
            latency_ms, cache_hits, cache_misses
        """
        t_start  = time.time()
        original = agent_a_result["original"]
        masked   = agent_a_result["masked"]
        spans    = agent_a_result["spans"]

        encrypted_spans = [
            s["span"] for s in spans if s.get("action") == "encrypt"
        ]

        # ── Run 3 checks in parallel ──────────────────────────────────────────
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(self._check_leakage,       original, masked):       "leakage",
                executor.submit(self._check_fluency,       masked):                 "fluency",
                executor.submit(self._check_overencryption,original, masked,
                                encrypted_spans):                                   "overencryption",
            }

            results = {}
            for future in as_completed(futures):
                key = futures[future]
                try:
                    results[key] = future.result()
                except Exception as e:
                    print(f"[AgentB] Check '{key}' failed: {e}")
                    results[key] = self._default_result(key)

        leakage       = results.get("leakage",       self._default_result("leakage"))
        fluency       = results.get("fluency",        self._default_result("fluency"))
        overencryption= results.get("overencryption", self._default_result("overencryption"))

        # ── Compute rewards ───────────────────────────────────────────────────
        reward, breakdown = self._compute_reward(
            agent_a_result, leakage, fluency, overencryption
        )

        # ── Update pattern memory ─────────────────────────────────────────────
        self._update_memory(spans, leakage, overencryption)

        latency_ms = (time.time() - t_start) * 1000

        return {
            "is_clean":         leakage["is_clean"],
            "is_fluent":        fluency["is_fluent"],
            "is_accurate":      overencryption["is_accurate"],
            "leaked_spans":     leakage["leaked_spans"],
            "over_encrypted":   overencryption["over_encrypted"],
            "fluency_score":    fluency["fluency_score"],
            "leakage_confidence": leakage["confidence"],
            "total_reward":     round(reward, 3),
            "reward_breakdown": breakdown,
            "latency_ms":       round(latency_ms, 2),
        }

    # ── Parallel Checks ───────────────────────────────────────────────────────

    def _check_leakage(self, original: str, masked: str) -> dict:
        """Check if any PHI leaked through into the masked sentence."""
        prompt = VALIDATE_PROMPT.format(original=original, masked=masked)
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0}
            )
            raw  = response["message"]["content"].strip()
            raw  = re.sub(r"```json|```", "", raw).strip()
            raw  = re.sub(r'[\x00-\x1f\x7f]', ' ', raw)
            data = json.loads(raw)
            return {
                "leaked_spans": data.get("leaked_spans", []),
                "is_clean":     data.get("is_clean", True),
                "confidence":   data.get("confidence", 0.5),
            }
        except Exception as e:
            print(f"[AgentB] Leakage check error: {e}")
            return self._default_result("leakage")

    def _check_fluency(self, masked: str) -> dict:
        """Check if masked sentence is semantically fluent."""
        prompt = FLUENCY_PROMPT.format(masked=masked)
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0}
            )
            raw  = response["message"]["content"].strip()
            raw  = re.sub(r"```json|```", "", raw).strip()
            raw  = re.sub(r'[\x00-\x1f\x7f]', ' ', raw)
            data = json.loads(raw)
            return {
                "is_fluent":     data.get("is_fluent", True),
                "fluency_score": data.get("fluency_score", 0.5),
                "reason":        data.get("reason", ""),
            }
        except Exception as e:
            print(f"[AgentB] Fluency check error: {e}")
            return self._default_result("fluency")

    def _check_overencryption(self, original: str, masked: str,
                               encrypted_spans: list[str]) -> dict:
        """Check if benign text was unnecessarily encrypted."""
        if not encrypted_spans:
            return {"over_encrypted": [], "is_accurate": True}

        prompt = OVERENCRYPTION_PROMPT.format(
            original=original,
            masked=masked,
            encrypted_spans=json.dumps(encrypted_spans)
        )
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0}
            )
            raw  = response["message"]["content"].strip()
            raw  = re.sub(r"```json|```", "", raw).strip()
            raw  = re.sub(r'[\x00-\x1f\x7f]', ' ', raw)
            data = json.loads(raw)
            return {
                "over_encrypted": data.get("over_encrypted", []),
                "is_accurate":    data.get("is_accurate", True),
            }
        except Exception as e:
            print(f"[AgentB] Over-encryption check error: {e}")
            return self._default_result("overencryption")

    # ── Reward Computation ────────────────────────────────────────────────────

    def _compute_reward(self, agent_a_result: dict, leakage: dict,
                        fluency: dict, overencryption: dict) -> tuple[float, dict]:
        breakdown = {}
        total     = 0.0

        # Correct redaction (no leakage)
        if leakage["is_clean"]:
            breakdown["correct_redaction"] = REWARD_CORRECT_REDACTION
        else:
            penalty = REWARD_MISSED_PHI * len(leakage["leaked_spans"])
            breakdown["missed_phi"] = penalty

        # Over-encryption penalty
        if not overencryption["is_accurate"]:
            penalty = REWARD_OVER_ENCRYPTION * len(overencryption["over_encrypted"])
            breakdown["over_encryption"] = penalty

        # Cache hit bonus
        if agent_a_result["cache_hits"] > 0:
            breakdown["cache_hit_bonus"] = REWARD_CACHE_HIT * agent_a_result["cache_hits"]

        # Fluency / utility preserved
        if fluency["is_fluent"]:
            breakdown["utility_preserved"] = REWARD_UTILITY_PRESERVED

        total = sum(breakdown.values())
        return total, breakdown

    # ── Memory Update ─────────────────────────────────────────────────────────

    def _update_memory(self, spans: list[dict], leakage: dict,
                       overencryption: dict):
        """
        Reward correct pattern hits, penalize patterns that missed or over-encrypted.
        """
        leaked_texts        = set(leakage.get("leaked_spans", []))
        over_encrypted_texts= set(overencryption.get("over_encrypted", []))

        for span_info in spans:
            label = span_info.get("label")
            if not label:
                continue

            span = span_info.get("span", "")

            if span in leaked_texts:
                # Pattern missed this — penalize
                self.memory.update_hit(label, correct=False)
            elif span in over_encrypted_texts:
                # Pattern over-fired — penalize
                self.memory.update_hit(label, correct=False)
            else:
                # Correct detection — reward
                self.memory.update_hit(label, correct=True)

        # Prune low-confidence patterns periodically
        self.memory.prune()

    # ── Defaults ─────────────────────────────────────────────────────────────

    def _default_result(self, check_type: str) -> dict:
        defaults = {
            "leakage":        {"leaked_spans": [], "is_clean": True,  "confidence": 0.5},
            "fluency":        {"is_fluent": True, "fluency_score": 0.5, "reason": "unknown"},
            "overencryption": {"over_encrypted": [], "is_accurate": True},
        }
        return defaults.get(check_type, {})
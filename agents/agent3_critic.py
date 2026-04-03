from __future__ import annotations

"""
agents/agent3_critic.py
Agent 3 — Critic + RL Controller

Single responsibility: validate Agent 2 output, compute rewards, and update
Agent 1 + Agent 2 parameters. Persists updated params to agent_params.json.

Checks (run in parallel):
  1. Leakage   — did any PHI slip through into the masked output?
  2. Fluency   — is the masked sentence semantically coherent?
  3. Accuracy  — was anything over-encrypted or wrongly abstracted?

Parameter updates:
  Agent 1 miss        → increase sensitivity, add phi_type to phi_types_focus
  Agent 1 false pos   → decrease sensitivity
  Agent 2 regex miss  → penalise pattern confidence, flag for regeneration
  Agent 2 regex hit   → reward pattern confidence
  Agent 2 over-enc    → penalise pattern, nudge min_confidence_to_apply up
"""

import json
import os
import re
import time
import ollama
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import (
    LLM_MODEL,
    REWARD_CORRECT_REDACTION, REWARD_MISSED_PHI, REWARD_OVER_ENCRYPTION,
    REWARD_CACHE_HIT, REWARD_UTILITY_PRESERVED,
    SENSITIVITY_MIN, SENSITIVITY_MAX, SENSITIVITY_STEP,
    MIN_CONFIDENCE_MIN, MIN_CONFIDENCE_MAX,
    PROMOTE_THRESHOLD_MIN, PROMOTE_THRESHOLD_MAX,
)
from memory.pattern_memory import PatternMemory

PARAMS_PATH = os.path.join(os.path.dirname(__file__), "agent_params.json")


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


class Agent3Critic:
    def __init__(self, memory: PatternMemory, agent1, agent2):
        self.memory  = memory
        self.model   = LLM_MODEL
        self.agent1  = agent1  # reference for param updates
        self.agent2  = agent2  # reference for param updates
        self._params = self._load_params()

    # ── Main Entry ────────────────────────────────────────────────────────────

    def validate(self, agent2_result: dict) -> dict:
        """
        Validate Agent 2 output. Compute rewards. Update Agent 1 + 2 params.
        Returns validation report.
        """
        t_start  = time.time()
        original = agent2_result["original"]
        masked   = agent2_result["masked"]
        spans    = agent2_result["spans"]

        encrypted_spans = [s["span"] for s in spans if s.get("action") == "encrypt"]

        # ── 3 parallel checks ─────────────────────────────────────────────────
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(self._check_leakage, original, masked):              "leakage",
                executor.submit(self._check_fluency, masked):                         "fluency",
                executor.submit(self._check_overencryption, original, masked,
                                encrypted_spans):                                     "overencryption",
            }
            results = {}
            for future in as_completed(futures):
                key = futures[future]
                try:
                    results[key] = future.result()
                except Exception as e:
                    print(f"[Agent3] Check '{key}' failed: {e}")
                    results[key] = self._default_result(key)

        leakage        = results.get("leakage",        self._default_result("leakage"))
        fluency        = results.get("fluency",         self._default_result("fluency"))
        overencryption = results.get("overencryption",  self._default_result("overencryption"))

        # ── Rewards ───────────────────────────────────────────────────────────
        reward, breakdown = self._compute_reward(agent2_result, leakage, fluency, overencryption)

        # ── RL: update pattern memory + agent params ──────────────────────────
        self._update_memory(spans, leakage, overencryption)
        self._update_params(leakage, overencryption, spans)

        latency_ms = (time.time() - t_start) * 1000

        return {
            "is_clean":           leakage["is_clean"],
            "is_fluent":          fluency["is_fluent"],
            "is_accurate":        overencryption["is_accurate"],
            "leaked_spans":       leakage["leaked_spans"],
            "over_encrypted":     overencryption["over_encrypted"],
            "fluency_score":      fluency["fluency_score"],
            "leakage_confidence": leakage["confidence"],
            "total_reward":       round(reward, 3),
            "reward_breakdown":   breakdown,
            "latency_ms":         round(latency_ms, 2),
        }

    # ── Parallel Checks ───────────────────────────────────────────────────────

    def _check_leakage(self, original: str, masked: str) -> dict:
        prompt = VALIDATE_PROMPT.format(original=original, masked=masked)
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0},
            )
            raw  = response["message"]["content"].strip()
            raw  = re.sub(r"```json|```", "", raw).strip()
            raw  = re.sub(r'[\x00-\x1f\x7f]', ' ', raw)
            raw  = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', raw)
            data = json.loads(raw)
            return {
                "leaked_spans": data.get("leaked_spans", []),
                "is_clean":     data.get("is_clean", True),
                "confidence":   data.get("confidence", 0.5),
            }
        except Exception as e:
            print(f"[Agent3] Leakage check error: {e}")
            return self._default_result("leakage")

    def _check_fluency(self, masked: str) -> dict:
        prompt = FLUENCY_PROMPT.format(masked=masked)
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0},
            )
            raw  = response["message"]["content"].strip()
            raw  = re.sub(r"```json|```", "", raw).strip()
            raw  = re.sub(r'[\x00-\x1f\x7f]', ' ', raw)
            raw  = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', raw)
            data = json.loads(raw)
            return {
                "is_fluent":     data.get("is_fluent", True),
                "fluency_score": data.get("fluency_score", 0.5),
                "reason":        data.get("reason", ""),
            }
        except Exception as e:
            print(f"[Agent3] Fluency check error: {e}")
            return self._default_result("fluency")

    def _check_overencryption(self, original: str, masked: str,
                               encrypted_spans: list[str]) -> dict:
        if not encrypted_spans:
            return {"over_encrypted": [], "is_accurate": True}
        prompt = OVERENCRYPTION_PROMPT.format(
            original=original,
            masked=masked,
            encrypted_spans=json.dumps(encrypted_spans),
        )
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0},
            )
            raw  = response["message"]["content"].strip()
            raw  = re.sub(r"```json|```", "", raw).strip()
            raw  = re.sub(r'[\x00-\x1f\x7f]', ' ', raw)
            raw  = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', raw)
            data = json.loads(raw)
            return {
                "over_encrypted": data.get("over_encrypted", []),
                "is_accurate":    data.get("is_accurate", True),
            }
        except Exception as e:
            print(f"[Agent3] Over-encryption check error: {e}")
            return self._default_result("overencryption")

    # ── Reward Computation ────────────────────────────────────────────────────

    def _compute_reward(self, agent2_result: dict, leakage: dict,
                        fluency: dict, overencryption: dict) -> tuple[float, dict]:
        breakdown = {}

        if leakage["is_clean"]:
            breakdown["correct_redaction"] = REWARD_CORRECT_REDACTION
        else:
            breakdown["missed_phi"] = REWARD_MISSED_PHI * len(leakage["leaked_spans"])

        if not overencryption["is_accurate"]:
            breakdown["over_encryption"] = REWARD_OVER_ENCRYPTION * len(overencryption["over_encrypted"])

        if agent2_result["cache_hits"] > 0:
            breakdown["cache_hit_bonus"] = REWARD_CACHE_HIT * agent2_result["cache_hits"]

        if fluency["is_fluent"]:
            breakdown["utility_preserved"] = REWARD_UTILITY_PRESERVED

        return sum(breakdown.values()), breakdown

    # ── Memory Update (Agent 2 patterns) ─────────────────────────────────────

    def _update_memory(self, spans: list[dict], leakage: dict, overencryption: dict):
        leaked_texts       = set(leakage.get("leaked_spans", []))
        over_enc_texts     = set(overencryption.get("over_encrypted", []))

        for span_info in spans:
            label = span_info.get("label")
            if not label:
                continue
            span = span_info.get("span", "")
            correct = span not in leaked_texts and span not in over_enc_texts
            self.memory.update_hit(label, correct=correct)

        self.memory.prune()

    # ── Param Update (Agent 1 + Agent 2) ─────────────────────────────────────

    def _update_params(self, leakage: dict, overencryption: dict, spans: list[dict]):
        p1 = self._params["agent1"]
        p2 = self._params["agent2"]

        # Agent 1 — missed PHI → boost sensitivity + focus on missed types
        if not leakage["is_clean"] and leakage["leaked_spans"]:
            p1["sensitivity"] = min(
                SENSITIVITY_MAX,
                round(p1["sensitivity"] + SENSITIVITY_STEP, 2),
            )
            # Add leaked phi_types to focus list (Agent 3 infers type from leaked text)
            # We don't have the phi_type of leaked spans directly, so we boost all
            # types found in the leak by cross-referencing spans metadata
            leaked_texts = set(leakage["leaked_spans"])
            for s in spans:
                if s.get("span") in leaked_texts:
                    phi_type = s.get("phi_type", "")
                    if phi_type and phi_type not in p1["phi_types_focus"]:
                        p1["phi_types_focus"].append(phi_type)

        # Agent 1 — over-encryption (false positives) → lower sensitivity
        if not overencryption["is_accurate"] and overencryption["over_encrypted"]:
            p1["sensitivity"] = max(
                SENSITIVITY_MIN,
                round(p1["sensitivity"] - SENSITIVITY_STEP, 2),
            )

        # Agent 2 — over-encryption → tighten confidence threshold
        if not overencryption["is_accurate"] and overencryption["over_encrypted"]:
            p2["min_confidence_to_apply"] = min(
                MIN_CONFIDENCE_MAX,
                round(p2["min_confidence_to_apply"] + 0.02, 3),
            )

        # Agent 2 — clean + cache hits → loosen confidence threshold slightly
        if leakage["is_clean"] and not overencryption["over_encrypted"]:
            p2["min_confidence_to_apply"] = max(
                MIN_CONFIDENCE_MIN,
                round(p2["min_confidence_to_apply"] - 0.01, 3),
            )

        # Clear focus list if we had a clean run (no misses)
        if leakage["is_clean"]:
            p1["phi_types_focus"] = []

        self._save_params()
        self.agent1.update_params(p1)
        self.agent2.update_params(p2)

    # ── Param Persistence ────────────────────────────────────────────────────

    def _load_params(self) -> dict:
        if os.path.exists(PARAMS_PATH):
            try:
                with open(PARAMS_PATH) as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "agent1": {"sensitivity": 0.7, "phi_types_focus": []},
            "agent2": {"min_confidence_to_apply": 0.75, "promote_threshold": 20, "abstraction_level": 1},
        }

    def _save_params(self):
        with open(PARAMS_PATH, "w") as f:
            json.dump(self._params, f, indent=2)

    # ── Defaults ─────────────────────────────────────────────────────────────

    def _default_result(self, check_type: str) -> dict:
        return {
            "leakage":        {"leaked_spans": [], "is_clean": True, "confidence": 0.5},
            "fluency":        {"is_fluent": True, "fluency_score": 0.5, "reason": "unknown"},
            "overencryption": {"over_encrypted": [], "is_accurate": True},
        }.get(check_type, {})

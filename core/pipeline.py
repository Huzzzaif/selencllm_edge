from __future__ import annotations

"""
core/pipeline.py
SELENCLLM Edge — Main Orchestrator

Per-sentence flow:
  Agent 1 detects PHI spans (sensitivity param from Agent 3)
  Agent 2 masks sentence   (confidence/promote params from Agent 3)
  Agent 3 validates + updates Agent 1 & 2 params async

Exposes: process(), process_batch(), process_async(), decrypt(), stats()
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, Future

from encryption.chacha import ChaChaEncryptor
from memory.pattern_memory import PatternMemory
from memory.seed_patterns import seed_memory
from agents.agent1_detector import Agent1Detector
from agents.agent2_pattern import Agent2PatternMasker
from agents.agent3_critic import Agent3Critic

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "agents", "agent_params.json")
_DEFAULT_PARAMS = {
    "agent1": {"sensitivity": 0.7, "phi_types_focus": []},
    "agent2": {"min_confidence_to_apply": 0.75, "promote_threshold": 20, "abstraction_level": 1},
}


def _load_params() -> dict:
    path = os.path.normpath(_PARAMS_PATH)
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return _DEFAULT_PARAMS


class SELENCLLMPipeline:
    def __init__(self, key: bytes = None):
        """
        key — optional ChaCha20 key bytes for session persistence.
              If None, a new key is generated per instantiation.
        """
        params = _load_params()

        self.encryptor = ChaChaEncryptor(key=key)
        self.memory    = PatternMemory()
        seed_memory(self.memory)

        self.agent1 = Agent1Detector(params=params["agent1"])
        self.agent2 = Agent2PatternMasker(
            memory=self.memory,
            encryptor=self.encryptor,
            params=params["agent2"],
        )
        self.agent3 = Agent3Critic(
            memory=self.memory,
            agent1=self.agent1,
            agent2=self.agent2,
        )

        self._stats = {
            "total_sentences":    0,
            "total_spans":        0,
            "total_encrypted":    0,
            "total_abstracted":   0,
            "total_cache_hits":   0,
            "total_cache_misses": 0,
            "total_reward":       0.0,
            "missed_phi_count":   0,
            "avg_latency_1_ms":   0.0,
            "avg_latency_2_ms":   0.0,
            "avg_latency_3_ms":   0.0,
        }

    # ── Main Entry ────────────────────────────────────────────────────────────

    def process(self, sentence: str, validate: bool = True) -> dict:
        """
        Process a single sentence.
        Agent 1 → Agent 2 → (Agent 3 synchronously if validate=True).
        """
        t1_start = time.time()
        spans = self.agent1.detect(sentence)
        latency_1 = (time.time() - t1_start) * 1000

        result2 = self.agent2.process(sentence, spans)
        result2["latency_1_ms"] = round(latency_1, 2)

        result3 = None
        if validate:
            result3 = self.agent3.validate(result2)

        self._update_stats(result2, result3)
        return self._build_output(result2, result3)

    def process_batch(self, sentences: list[str],
                      validate: bool = True,
                      max_workers: int = 1) -> list[dict]:
        """Process a list of sentences sequentially or in parallel."""
        if max_workers == 1:
            return [self.process(s, validate=validate) for s in sentences]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.process, s, validate) for s in sentences]
            return [f.result() for f in futures]

    def process_async(self, sentence: str) -> tuple[dict, Future]:
        """
        Fire-and-forget: returns Agent 2 result immediately,
        Agent 3 validation runs in background.
        """
        spans   = self.agent1.detect(sentence)
        result2 = self.agent2.process(sentence, spans)

        executor = ThreadPoolExecutor(max_workers=1)
        future3  = executor.submit(self.agent3.validate, result2)

        return result2, future3

    # ── Decrypt ───────────────────────────────────────────────────────────────

    def decrypt(self, masked_sentence: str, vault_snapshot: dict) -> str:
        return self.encryptor.decrypt_sentence(masked_sentence, vault_snapshot)

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        s = self._stats.copy()
        s["memory"]             = self.memory.stats()
        s["encryption_key_b64"] = self.encryptor.export_key()
        s["agent1_params"]      = {
            "sensitivity":     self.agent1.sensitivity,
            "phi_types_focus": self.agent1.phi_types_focus,
        }
        s["agent2_params"] = {
            "min_confidence_to_apply": self.agent2.min_confidence_to_apply,
            "promote_threshold":       self.agent2.promote_threshold,
            "abstraction_level":       self.agent2.abstraction_level,
        }
        return s

    def reset_stats(self):
        for k in self._stats:
            self._stats[k] = 0.0 if isinstance(self._stats[k], float) else 0

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_output(self, result2: dict, result3: dict | None) -> dict:
        output = {
            "original":         result2["original"],
            "masked":           result2["masked"],
            "vault_snapshot":   result2["vault_snapshot"],
            "spans":            result2["spans"],
            "span_count":       len(result2["spans"]),
            "latency_1_ms":     result2.get("latency_1_ms", 0.0),
            "latency_a_ms":     result2["latency_ms"],   # Agent 2 masking latency
            "cache_hits":       result2["cache_hits"],
            "cache_misses":     result2["cache_misses"],
            "encrypted_count":  sum(1 for s in result2["spans"] if s.get("action") == "encrypt"),
            "abstracted_count": sum(1 for s in result2["spans"] if s.get("action") == "abstract"),
        }
        if result3:
            output["validation"] = {
                "is_clean":           result3["is_clean"],
                "is_fluent":          result3["is_fluent"],
                "is_accurate":        result3["is_accurate"],
                "leaked_spans":       result3["leaked_spans"],
                "over_encrypted":     result3["over_encrypted"],
                "fluency_score":      result3["fluency_score"],
                "leakage_confidence": result3["leakage_confidence"],
                "total_reward":       result3["total_reward"],
                "reward_breakdown":   result3["reward_breakdown"],
                "latency_b_ms":       result3["latency_ms"],
            }
        return output

    def _update_stats(self, result2: dict, result3: dict | None):
        s = self._stats
        n = s["total_sentences"] + 1

        s["total_sentences"]    += 1
        s["total_spans"]        += len(result2["spans"])
        s["total_cache_hits"]   += result2["cache_hits"]
        s["total_cache_misses"] += result2["cache_misses"]
        s["total_encrypted"]    += sum(1 for sp in result2["spans"] if sp.get("action") == "encrypt")
        s["total_abstracted"]   += sum(1 for sp in result2["spans"] if sp.get("action") == "abstract")

        s["avg_latency_1_ms"] = (s["avg_latency_1_ms"] * (n - 1) + result2.get("latency_1_ms", 0)) / n
        s["avg_latency_2_ms"] = (s["avg_latency_2_ms"] * (n - 1) + result2["latency_ms"]) / n

        if result3:
            s["total_reward"]     += result3["total_reward"]
            s["missed_phi_count"] += len(result3["leaked_spans"])
            s["avg_latency_3_ms"]  = (s["avg_latency_3_ms"] * (n - 1) + result3["latency_ms"]) / n

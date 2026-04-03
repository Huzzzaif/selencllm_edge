"""
memory/pattern_memory.py
Persistent adaptive regex + abstraction cache with RL-style promote/prune.
"""

import json
import re
import time
import os
from config import (
    MEMORY_PATH, HIT_PROMOTE_THRESHOLD,
    CONFIDENCE_PRUNE_BELOW, MAX_MEMORY_ENTRIES
)


class PatternMemory:
    def __init__(self, path: str = MEMORY_PATH):
        self.path = path
        self.memory: dict[str, dict] = {}
        self._load()

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def add(self, label: str, regex: str, phi_type: str, abstraction_role: str = None):
        """Add or update a pattern entry."""
        if label in self.memory:
            return  # already exists, use update_hit to modify

        self.memory[label] = {
            "regex":             regex,
            "phi_type":          phi_type,          # PERSON, SSN, LOCATION, etc.
            "abstraction_role":  abstraction_role,   # "the physician", "an urban area"
            "hits":              0,
            "misses":            0,
            "confidence":        0.5,               # starts neutral
            "promoted":          False,              # True = skip LLM, apply directly
            "last_updated":      time.strftime("%Y-%m-%d"),
        }
        self._save()

    def match(self, text: str) -> list[dict]:
        """
        Run all patterns against text.
        Returns list of matches: [{label, span, phi_type, abstraction_role, promoted}]
        Promoted patterns checked first.
        """
        results = []
        sorted_entries = sorted(
            self.memory.items(),
            key=lambda x: (not x[1]["promoted"], -x[1]["confidence"])
        )

        for label, entry in sorted_entries:
            try:
                for m in re.finditer(entry["regex"], text):
                    results.append({
                        "label":            label,
                        "span":             m.group(),
                        "start":            m.start(),
                        "end":              m.end(),
                        "phi_type":         entry["phi_type"],
                        "abstraction_role": entry.get("abstraction_role"),
                        "promoted":         entry["promoted"],
                        "confidence":       entry["confidence"],
                    })
            except re.error:
                continue  # bad regex — will be pruned eventually

        return results

    def update_hit(self, label: str, correct: bool):
        """Called by Agent B feedback loop."""
        if label not in self.memory:
            return

        entry = self.memory[label]
        if correct:
            entry["hits"] += 1
        else:
            entry["misses"] += 1

        total = entry["hits"] + entry["misses"]
        entry["confidence"]   = entry["hits"] / total if total > 0 else 0.5
        entry["last_updated"] = time.strftime("%Y-%m-%d")

        # Auto-promote
        if entry["hits"] >= HIT_PROMOTE_THRESHOLD and entry["confidence"] > 0.75:
            entry["promoted"] = True

        self._save()

    def update_abstraction(self, label: str, abstraction_role: str):
        """Update the semantic abstraction role for a pattern."""
        if label in self.memory:
            self.memory[label]["abstraction_role"] = abstraction_role
            self._save()

    # ── RL Maintenance ────────────────────────────────────────────────────────

    def prune(self):
        """Remove low-confidence and excess entries."""
        # Remove low confidence
        to_remove = [
            k for k, v in self.memory.items()
            if v["confidence"] < CONFIDENCE_PRUNE_BELOW and v["hits"] + v["misses"] > 10
        ]
        for k in to_remove:
            del self.memory[k]

        # Cap size — remove lowest confidence unpromoted entries
        if len(self.memory) > MAX_MEMORY_ENTRIES:
            unpromoted = sorted(
                [(k, v) for k, v in self.memory.items() if not v["promoted"]],
                key=lambda x: x[1]["confidence"]
            )
            excess = len(self.memory) - MAX_MEMORY_ENTRIES
            for k, _ in unpromoted[:excess]:
                del self.memory[k]

        self._save()

    def stats(self) -> dict:
        total     = len(self.memory)
        promoted  = sum(1 for v in self.memory.values() if v["promoted"])
        avg_conf  = (
            sum(v["confidence"] for v in self.memory.values()) / total
            if total > 0 else 0
        )
        return {"total": total, "promoted": promoted, "avg_confidence": round(avg_conf, 3)}

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.memory, f, indent=2)

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                self.memory = json.load(f)

    def __repr__(self):
        s = self.stats()
        return f"PatternMemory(total={s['total']}, promoted={s['promoted']}, avg_conf={s['avg_confidence']})"
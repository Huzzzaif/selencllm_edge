"""
eval/reconstruction_attack.py
SELENCLLM Edge — Reconstruction Attack Simulation

Simulates an adversary who receives anonymized output and prompts an on-device
LLM (llama3.1:8b via ollama) to recover original PII.

Attack success metric: any attacker guess shares tokens with a ground-truth span.

Usage:
    python eval/reconstruction_attack.py --n 100
    python eval/reconstruction_attack.py --n 5 --systems regex,sensecllm
"""

import os
import sys
import json
import csv
import re
import time
import argparse

import ollama
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.runner import load_ai4privacy
from eval.benchmark import RegexBaseline, SpacyBaseline, PresidioBaseline, SELENCLLMBaseline
from encryption.chacha import ChaChaEncryptor
from config import LLM_MODEL

TABLES_DIR = os.path.join(os.path.dirname(__file__), "..", "paper", "tables")
os.makedirs(TABLES_DIR, exist_ok=True)

ATTACK_PROMPT = """\
You are an adversary trying to recover sensitive information from anonymized text.
Given the anonymized text below, identify and list what you think the original \
sensitive information was (names, locations, ID numbers, medical conditions, etc.)
Be specific — guess actual values where possible.

Anonymized text: {masked_text}

List each piece of sensitive information you can recover or infer.
Respond ONLY with a JSON array of strings. No explanation.
["guess1", "guess2", ...]"""


# ── Attack logic ──────────────────────────────────────────────────────────────

def attack_success(guess: str, original_span: str) -> bool:
    guess_tokens = set(guess.lower().split())
    orig_tokens  = set(original_span.lower().split())
    return bool(guess_tokens & orig_tokens)


def run_attack(masked_text: str) -> list[str]:
    """Ask the LLM to reconstruct PII from masked text. Returns list of guesses."""
    prompt = ATTACK_PROMPT.format(masked_text=masked_text)
    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3},   # slight temp for realistic attacker
        )
        raw = response["message"]["content"].strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        raw = re.sub(r'[\x00-\x1f\x7f]', ' ', raw)
        raw = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', raw)
        # Extract JSON array even if model wraps it in prose
        match = re.search(r'\[.*?\]', raw, re.DOTALL)
        if match:
            raw = match.group(0)
        guesses = json.loads(raw)
        if isinstance(guesses, list):
            return [str(g) for g in guesses if g]
        return []
    except Exception as e:
        print(f"  [Attack] LLM error: {e}")
        return []


# ── Full-encryption baseline ──────────────────────────────────────────────────

class FullEncryptionBaseline:
    """Replaces every ground-truth span with an opaque ChaCha20 token."""
    name = "full_encryption"

    def __init__(self):
        self._encryptor = ChaChaEncryptor()

    def process(self, sentence: str, gt_spans: list[str]) -> dict:
        t = time.time()
        masked = sentence
        # Sort by position descending to avoid offset drift
        positioned = sorted(
            [(masked.find(sp), sp) for sp in gt_spans if sp and sp in masked],
            reverse=True
        )
        pred_spans = []
        for idx, span in positioned:
            if idx == -1:
                continue
            token, _ = self._encryptor.encrypt_span(span)
            masked = masked[:idx] + token + masked[idx + len(span):]
            pred_spans.append({"span": span, "phi_type": "UNKNOWN"})
        return {
            "masked":     masked,
            "pred_spans": pred_spans,
            "latency_ms": (time.time() - t) * 1000,
            "cache_hits": 0,
        }


# ── System builder ────────────────────────────────────────────────────────────

ALL_SYSTEMS = ["regex", "spacy", "presidio", "sensecllm", "full_encryption"]

def build_attack_systems(selected: list[str]) -> list:
    systems = []
    for name in selected:
        if name == "regex":
            systems.append(RegexBaseline())
        elif name == "spacy":
            systems.append(SpacyBaseline())
        elif name == "presidio":
            systems.append(PresidioBaseline())
        elif name == "sensecllm":
            systems.append(SELENCLLMBaseline("sensecllm", fresh_memory=False))
        elif name == "full_encryption":
            systems.append(FullEncryptionBaseline())
    return systems


# ── Main runner ───────────────────────────────────────────────────────────────

def run_reconstruction_attack(samples: list[dict], systems: list) -> dict:
    # Per-system accumulators
    totals    = {s.name: {"total_spans": 0, "recovered_spans": 0} for s in systems}
    per_sample = []   # detailed per-sample records

    print(f"\nRunning reconstruction attack on {len(samples)} samples "
          f"× {len(systems)} systems...\n")

    for i, sample in enumerate(tqdm(samples, desc="Attack")):
        text     = sample["text"]
        gt_spans = [s["span"] for s in sample["phi_spans"] if s.get("span")]

        sample_record = {
            "sample_id": i,
            "original":  text,
            "gt_spans":  gt_spans,
            "systems":   {},
        }

        for sys in systems:
            try:
                out = sys.process(text, gt_spans)
            except Exception as e:
                print(f"  [{sys.name}] Process error on sample {i}: {e}")
                out = {"masked": text, "pred_spans": [], "latency_ms": 0, "cache_hits": 0}

            masked = out["masked"]
            guesses = run_attack(masked)

            # Count successes
            recovered = 0
            for gt_span in gt_spans:
                if any(attack_success(g, gt_span) for g in guesses):
                    recovered += 1

            totals[sys.name]["total_spans"]     += len(gt_spans)
            totals[sys.name]["recovered_spans"] += recovered

            sample_record["systems"][sys.name] = {
                "masked":       masked,
                "guesses":      guesses,
                "recovered":    recovered,
                "total_gt":     len(gt_spans),
            }

        per_sample.append(sample_record)

    # Compute final rates
    results = []
    for sys in systems:
        t = totals[sys.name]
        total     = t["total_spans"]
        recovered = t["recovered_spans"]
        rate      = recovered / total if total > 0 else 0.0
        results.append({
            "system":              sys.name,
            "attack_success_rate": round(rate, 4),
            "total_spans":         total,
            "recovered_spans":     recovered,
        })

    return {"summary": results, "per_sample": per_sample}


# ── Save results ──────────────────────────────────────────────────────────────

def save_results(results: dict):
    # CSV summary
    csv_path = os.path.join(TABLES_DIR, "reconstruction_attack.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["system", "attack_success_rate", "total_spans", "recovered_spans"]
        )
        writer.writeheader()
        writer.writerows(results["summary"])
    print(f"  Saved {csv_path}")

    # JSON full results (skip per_sample to keep file size reasonable — just summary + first 20)
    json_path = os.path.join(TABLES_DIR, "reconstruction_attack.json")
    export = {
        "summary":    results["summary"],
        "per_sample": results["per_sample"][:20],   # first 20 samples for inspection
    }
    with open(json_path, "w") as f:
        json.dump(export, f, indent=2)
    print(f"  Saved {json_path}")


# ── Console summary ───────────────────────────────────────────────────────────

DISPLAY_NAMES = {
    "regex":            "Regex [REDACTED]",
    "spacy":            "spaCy [REDACTED]",
    "presidio":         "Presidio",
    "sensecllm":        "SenseCLLM",
    "full_encryption":  "Full Encryption",
}

def print_summary(results: dict):
    rows = results["summary"]
    print("\n" + "═" * 46)
    print("  RECONSTRUCTION ATTACK RESULTS")
    print("═" * 46)
    print(f"  {'System':<24} {'Attack Success Rate':>18}")
    print("─" * 46)
    for r in rows:
        label = DISPLAY_NAMES.get(r["system"], r["system"])
        print(f"  {label:<24} {r['attack_success_rate']:>18.4f}")
    print("═" * 46 + "\n")


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SELENCLLM Reconstruction Attack")
    parser.add_argument("--n",       type=int, default=100,
                        help="Number of samples to attack (default: 100)")
    parser.add_argument("--systems", type=str,
                        default="regex,spacy,presidio,sensecllm,full_encryption",
                        help="Comma-separated systems to attack")
    args = parser.parse_args()

    selected = [s.strip() for s in args.systems.split(",")]
    print(f"Systems: {selected}  |  Samples: {args.n}")

    samples = load_ai4privacy(n=args.n)
    systems = build_attack_systems(selected)
    results = run_reconstruction_attack(samples, systems)

    print("\nSaving results...")
    save_results(results)
    print_summary(results)

"""
eval/benchmark.py
SELENCLLM Edge — Comparative Benchmark

Runs SELENCLLM against three baselines on ai4privacy and saves results
for the research paper to paper/tables/.

Usage:
    python eval/benchmark.py --n 500
    python eval/benchmark.py --n 100 --systems regex,spacy,cold
"""

import os
import sys
import json
import time
import shutil
import argparse
import csv
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.runner import load_ai4privacy, load_mtsamples, compute_utility_score

TABLES_DIR  = os.path.join(os.path.dirname(__file__), "..", "paper", "tables")
os.makedirs(TABLES_DIR, exist_ok=True)

MEMORY_PATH = os.path.join(os.path.dirname(__file__), "..", "memory", "pattern_memory.json")


# ── Overlap matching ──────────────────────────────────────────────────────────

def spans_overlap(pred: str, gt: str) -> bool:
    pred_tokens = set(pred.lower().split())
    gt_tokens   = set(gt.lower().split())
    return bool(pred_tokens & gt_tokens) or pred in gt or gt in pred


def compute_metrics_overlap(gt_spans: list[str], pred_spans: list[str]) -> dict:
    """Precision/recall/F1 using overlap matching."""
    if not gt_spans and not pred_spans:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 0, "fp": 0, "fn": 0}
    if not pred_spans:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": len(gt_spans)}
    if not gt_spans:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0, "tp": 0, "fp": len(pred_spans), "fn": 0}

    matched_gt  = set()
    matched_pred = set()
    for gi, g in enumerate(gt_spans):
        for pi, p in enumerate(pred_spans):
            if pi not in matched_pred and spans_overlap(p, g):
                matched_gt.add(gi)
                matched_pred.add(pi)
                break

    tp = len(matched_gt)
    fp = len(pred_spans) - len(matched_pred)
    fn = len(gt_spans)   - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def exact_leakage(original: str, masked: str, gt_spans: list[str]) -> bool:
    """True if any ground truth span appears verbatim in the masked output."""
    for span in gt_spans:
        if span and span in masked:
            return True
    return False


def quasi_id_risk(original: str, masked: str) -> float:
    """Average token overlap ratio between original and masked."""
    orig_toks   = original.lower().split()
    masked_toks = set(masked.lower().split())
    if not orig_toks:
        return 0.0
    return sum(1 for t in orig_toks if t in masked_toks) / len(orig_toks)


# ── Baselines ─────────────────────────────────────────────────────────────────

class RegexBaseline:
    """Applies only the seed regex patterns — no LLM."""
    name = "regex"

    def __init__(self):
        import re
        from memory.seed_patterns import SEED_PATTERNS
        from encryption.chacha import ChaChaEncryptor
        self._patterns  = [(label, re.compile(regex), phi_type)
                           for label, regex, phi_type, _ in SEED_PATTERNS]
        self._encryptor = ChaChaEncryptor()

    def process(self, sentence: str, gt_spans: list[str]) -> dict:
        t = time.time()
        pred_spans = []
        # Find all matches in original sentence (sorted rightmost first to preserve offsets)
        hits = []
        for label, pattern, phi_type in self._patterns:
            for m in pattern.finditer(sentence):
                pred_spans.append({"span": m.group(), "phi_type": phi_type})
                hits.append((m.start(), m.end(), m.group()))
        hits.sort(key=lambda x: x[0], reverse=True)

        masked = sentence
        seen = set()
        for start, end, span in hits:
            if span in seen:
                continue
            seen.add(span)
            token, _ = self._encryptor.encrypt_span(span)
            idx = masked.find(span)
            if idx != -1:
                masked = masked[:idx] + token + masked[idx + len(span):]
        return {
            "masked":      masked,
            "pred_spans":  pred_spans,
            "latency_ms":  (time.time() - t) * 1000,
            "cache_hits":  0,
        }


class SpacyBaseline:
    """Uses spaCy en_core_web_lg NER."""
    name = "spacy"
    _LABEL_MAP = {
        "PERSON": "PERSON", "ORG": "ORG", "GPE": "LOCATION",
        "LOC": "LOCATION", "DATE": "DATE", "TIME": "DATE",
        "CARDINAL": "AGE", "MONEY": "CREDIT_CARD",
    }

    def __init__(self):
        import spacy
        from encryption.chacha import ChaChaEncryptor
        self._nlp       = spacy.load("en_core_web_lg")
        self._encryptor = ChaChaEncryptor()

    def process(self, sentence: str, gt_spans: list[str]) -> dict:
        t = time.time()
        doc = self._nlp(sentence)
        pred_spans = []
        masked = sentence
        for ent in reversed(doc.ents):  # reversed preserves char offsets
            phi_type = self._LABEL_MAP.get(ent.label_, ent.label_)
            pred_spans.append({"span": ent.text, "phi_type": phi_type})
            token, _ = self._encryptor.encrypt_span(ent.text)
            masked = masked[:ent.start_char] + token + masked[ent.end_char:]
        return {
            "masked":     masked,
            "pred_spans": pred_spans,
            "latency_ms": (time.time() - t) * 1000,
            "cache_hits": 0,
        }


class PresidioBaseline:
    """Uses Microsoft Presidio analyzer + anonymizer."""
    name = "presidio"

    def __init__(self):
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine
        from encryption.chacha import ChaChaEncryptor
        self._analyzer   = AnalyzerEngine()
        self._anonymizer = AnonymizerEngine()
        self._encryptor  = ChaChaEncryptor()

    def process(self, sentence: str, gt_spans: list[str]) -> dict:
        t = time.time()
        try:
            results = self._analyzer.analyze(text=sentence, language="en")
            pred_spans = [
                {"span": sentence[r.start:r.end], "phi_type": r.entity_type}
                for r in results
            ]
            # Encrypt each detected span (rightmost first to preserve offsets)
            masked = sentence
            for r in sorted(results, key=lambda x: x.start, reverse=True):
                span = sentence[r.start:r.end]
                token, _ = self._encryptor.encrypt_span(span)
                masked = masked[:r.start] + token + masked[r.end:]
        except Exception as e:
            print(f"[Presidio] Error: {e}")
            pred_spans = []
            masked = sentence

        return {
            "masked":     masked,
            "pred_spans": pred_spans,
            "latency_ms": (time.time() - t) * 1000,
            "cache_hits": 0,
        }


class SELENCLLMBaseline:
    """Wraps the full 3-agent pipeline."""

    def __init__(self, name: str, fresh_memory: bool = False):
        self.name = name
        if fresh_memory:
            # Delete learned memory, keep only seed patterns
            if os.path.exists(MEMORY_PATH):
                os.remove(MEMORY_PATH)
        from core.pipeline import SELENCLLMPipeline
        self._pipeline = SELENCLLMPipeline()

    def process(self, sentence: str, gt_spans: list[str]) -> dict:
        t = time.time()
        result = self._pipeline.process(sentence, validate=False)
        return {
            "masked":     result["masked"],
            "pred_spans": result["spans"],
            "latency_ms": (time.time() - t) * 1000,
            "cache_hits": result["cache_hits"],
        }


# ── Per-type tracking ─────────────────────────────────────────────────────────

def update_per_type(per_type: dict, system: str,
                    gt_spans: list[dict], pred_spans: list[dict]):
    """Overlap-based per-type TP/FP/FN accumulation."""
    gt_by_type   = defaultdict(list)
    pred_by_type = defaultdict(list)
    for s in gt_spans:
        gt_by_type[s.get("phi_type", "UNKNOWN")].append(s["span"])
    for s in pred_spans:
        pred_by_type[s.get("phi_type", "UNKNOWN")].append(s["span"] if isinstance(s, dict) else s)

    all_types = set(list(gt_by_type.keys()) + list(pred_by_type.keys()))
    for phi_type in all_types:
        m = compute_metrics_overlap(gt_by_type[phi_type], pred_by_type[phi_type])
        per_type[(system, phi_type)]["tp"] += m["tp"]
        per_type[(system, phi_type)]["fp"] += m["fp"]
        per_type[(system, phi_type)]["fn"] += m["fn"]


# ── Main runner ───────────────────────────────────────────────────────────────

def build_systems(selected: list[str]) -> list:
    systems = []
    if "regex" in selected:
        systems.append(RegexBaseline())
    if "spacy" in selected:
        systems.append(SpacyBaseline())
    if "presidio" in selected:
        systems.append(PresidioBaseline())
    if "cold" in selected:
        systems.append(SELENCLLMBaseline("cold", fresh_memory=True))
    if "warm" in selected:
        systems.append(SELENCLLMBaseline("warm", fresh_memory=False))
    return systems


def run_benchmark(samples: list[dict], systems: list) -> dict:
    n = len(samples)
    system_names = [s.name for s in systems]

    # Accumulators
    records      = {s.name: [] for s in systems}
    per_type     = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    scale_windows = {s.name: [] for s in systems if s.name in ("cold", "warm")}

    print(f"\nRunning benchmark on {n} samples across {len(systems)} systems...\n")

    for i, sample in enumerate(tqdm(samples, desc="Benchmark")):
        text     = sample["text"]
        gt_spans = sample["phi_spans"]   # list of {span, phi_type}
        gt_texts = [s["span"] for s in gt_spans]

        for sys in systems:
            try:
                out = sys.process(text, gt_texts)
            except Exception as e:
                print(f"[{sys.name}] Sample {i} error: {e}")
                out = {"masked": text, "pred_spans": [], "latency_ms": 0.0, "cache_hits": 0}

            pred_spans = out["pred_spans"]
            pred_texts = [
                (s["span"] if isinstance(s, dict) else s) for s in pred_spans
            ]
            masked = out["masked"]

            m         = compute_metrics_overlap(gt_texts, pred_texts)
            leakage   = exact_leakage(text, masked, gt_texts)
            quasi     = quasi_id_risk(text, masked)
            utility   = compute_utility_score(text, masked)

            records[sys.name].append({
                "precision":   m["precision"],
                "recall":      m["recall"],
                "f1":          m["f1"],
                "latency_ms":  out["latency_ms"],
                "cache_hits":  out["cache_hits"],
                "leakage":     int(leakage),
                "quasi_risk":  quasi,
                "utility":     utility,
            })

            update_per_type(per_type, sys.name, gt_spans, pred_spans)

            # Scale windows for SELENCLLM variants
            if sys.name in scale_windows and (i + 1) % 100 == 0:
                window = records[sys.name][-100:]
                scale_windows[sys.name].append({
                    "step":           i + 1,
                    "f1":             round(np.mean([r["f1"] for r in window]), 4),
                    "latency_ms":     round(np.mean([r["latency_ms"] for r in window]), 2),
                    "cache_hit_rate": round(
                        sum(1 for r in window if r["cache_hits"] > 0) / len(window), 4
                    ),
                })

    return _compile_results(records, per_type, scale_windows, system_names)


def _compile_results(records: dict, per_type: dict,
                     scale_windows: dict, system_names: list) -> dict:
    detection  = []
    latency    = []
    security   = []
    per_type_rows = []
    scale_rows    = []

    for name in system_names:
        recs = records[name]
        if not recs:
            continue
        df = pd.DataFrame(recs)

        # Detection
        detection.append({
            "system":    name,
            "precision": round(df["precision"].mean(), 4),
            "recall":    round(df["recall"].mean(), 4),
            "f1":        round(df["f1"].mean(), 4),
        })

        # Latency
        latency.append({
            "system":   name,
            "mean_ms":  round(df["latency_ms"].mean(), 2),
            "p95_ms":   round(df["latency_ms"].quantile(0.95), 2),
            "min_ms":   round(df["latency_ms"].min(), 2),
        })

        # Security
        security.append({
            "system":         name,
            "exact_leakage":  round(df["leakage"].mean(), 4),
            "quasi_id_risk":  round(df["quasi_risk"].mean(), 4),
            "utility_score":  round(df["utility"].mean(), 4),
        })

        # Scale windows
        if name in scale_windows:
            for w in scale_windows[name]:
                scale_rows.append({"system": name, **w})

    # Per-type
    seen_combos = set()
    for (sysname, phi_type), counts in per_type.items():
        if sysname not in system_names:
            continue
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        per_type_rows.append({
            "system":    sysname,
            "phi_type":  phi_type,
            "precision": round(p, 4),
            "recall":    round(r, 4),
            "f1":        round(f1, 4),
        })

    return {
        "detection":  detection,
        "latency":    latency,
        "security":   security,
        "scale":      scale_rows,
        "per_type":   per_type_rows,
    }


# ── Save results ──────────────────────────────────────────────────────────────

def save_results(results: dict, tag: str = ""):
    """Save benchmark outputs. tag is appended to filenames (e.g. '_mtsamples')."""
    suffix = f"_{tag}" if tag else ""

    def write_csv(name: str, rows: list[dict]):
        if not rows:
            return
        path = os.path.join(TABLES_DIR, name)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Saved {path}")

    write_csv(f"benchmark_detection{suffix}.csv", results["detection"])
    write_csv(f"benchmark_latency{suffix}.csv",   results["latency"])
    write_csv(f"benchmark_security{suffix}.csv",  results["security"])
    write_csv(f"benchmark_scale{suffix}.csv",     results["scale"])
    write_csv(f"benchmark_per_type{suffix}.csv",  results["per_type"])

    summary_path = os.path.join(TABLES_DIR, f"benchmark_summary{suffix}.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved {summary_path}")


# ── Console summary table ─────────────────────────────────────────────────────

def print_summary(results: dict):
    det = {r["system"]: r for r in results["detection"]}
    lat = {r["system"]: r for r in results["latency"]}
    sec = {r["system"]: r for r in results["security"]}

    systems = list(det.keys())
    col_w   = 12

    def row(label, vals):
        return f"  {label:<22}" + "".join(f"{v:>{col_w}}" for v in vals)

    header = f"  {'Metric':<22}" + "".join(f"{s:>{col_w}}" for s in systems)
    div    = "  " + "-" * (22 + col_w * len(systems))

    print("\n" + "=" * (24 + col_w * len(systems)))
    print("  BENCHMARK SUMMARY")
    print("=" * (24 + col_w * len(systems)))
    print(header)
    print(div)
    print(row("Precision",       [f"{det[s]['precision']:.4f}" if s in det else "-" for s in systems]))
    print(row("Recall",          [f"{det[s]['recall']:.4f}"    if s in det else "-" for s in systems]))
    print(row("F1",              [f"{det[s]['f1']:.4f}"        if s in det else "-" for s in systems]))
    print(div)
    print(row("Mean Latency ms", [f"{lat[s]['mean_ms']:.1f}"   if s in lat else "-" for s in systems]))
    print(row("P95 Latency ms",  [f"{lat[s]['p95_ms']:.1f}"    if s in lat else "-" for s in systems]))
    print(div)
    print(row("Exact Leakage",   [f"{sec[s]['exact_leakage']:.4f}" if s in sec else "-" for s in systems]))
    print(row("Quasi-ID Risk",   [f"{sec[s]['quasi_id_risk']:.4f}" if s in sec else "-" for s in systems]))
    print(row("Utility Score",   [f"{sec[s]['utility_score']:.4f}" if s in sec else "-" for s in systems]))
    print("=" * (24 + col_w * len(systems)) + "\n")


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SELENCLLM Benchmark")
    parser.add_argument("--n",       type=int, default=500,
                        help="Number of samples per dataset (default: 500)")
    parser.add_argument("--systems", type=str, default="regex,spacy,presidio,cold,warm",
                        help="Comma-separated systems (regex,spacy,presidio,cold,warm)")
    parser.add_argument("--dataset", type=str, default="ai4privacy",
                        help="Dataset(s) to run: ai4privacy, mtsamples, or both (default: ai4privacy)")
    args = parser.parse_args()

    selected  = [s.strip() for s in args.systems.split(",")]
    datasets  = [d.strip() for d in args.dataset.split(",")]
    if "both" in datasets:
        datasets = ["ai4privacy", "mtsamples"]

    print(f"Systems: {selected}  |  Samples: {args.n}  |  Datasets: {datasets}")

    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"  Dataset: {dataset.upper()}")
        print(f"{'='*60}")

        if dataset == "ai4privacy":
            samples = load_ai4privacy(n=args.n)
            tag     = ""
        elif dataset == "mtsamples":
            samples = load_mtsamples(n=args.n)
            tag     = "mtsamples"
            print("  Note: MTSamples has no PHI annotations — "
                  "detection metrics (P/R/F1) are not available for this dataset.")
        else:
            print(f"  Unknown dataset '{dataset}', skipping.")
            continue

        # Rebuild systems per dataset run (resets memory state for cold)
        systems = build_systems(selected)
        results = run_benchmark(samples, systems)

        print(f"\nSaving results ({dataset})...")
        save_results(results, tag=tag)
        print_summary(results)

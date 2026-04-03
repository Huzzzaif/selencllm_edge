"""
eval/runner.py
SELENCLLM Edge — Evaluation Runner

Runs pipeline against ai4privacy dataset and produces:
- Precision, Recall, F1 for PHI detection
- Miss rate, false positive rate
- Latency breakdown (cache hit vs miss vs full LLM)
- Memory growth curve
- Utility preservation score (masked text vs original — length/token similarity)
- Per PHI-type breakdown
- Saves results to paper/tables/ as CSV + LaTeX
"""

import os
import sys
import time
import json
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pipeline import SELENCLLMPipeline
from config import EVAL_SAMPLE_SIZE, DATASET_NAME

# Output dirs
TABLES_DIR  = os.path.join(os.path.dirname(__file__), "..", "paper", "tables")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "paper", "figures")
os.makedirs(TABLES_DIR,  exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ── Dataset Loader ────────────────────────────────────────────────────────────

def load_ai4privacy(n: int = EVAL_SAMPLE_SIZE) -> list[dict]:
    """
    Load ai4privacy dataset samples.
    Returns list of {text, phi_spans: [str]} dicts.
    """
    print(f"[Eval] Loading {DATASET_NAME} ...")
    ds = load_dataset(DATASET_NAME, split="train", streaming=True)

    samples = []
    for row in ds:
        if len(samples) >= n:
            break

        text = row.get("source_text") or row.get("text") or ""
        if not text.strip():
            continue

        # Extract ground truth PHI spans from privacy_mask field
        phi_spans = []
        masks = row.get("privacy_mask", [])
        if isinstance(masks, list):
            for m in masks:
                if isinstance(m, dict):
                    val = m.get("value") or m.get("text") or m.get("span", "")
                    if val and val in text:
                        phi_spans.append({
                            "span":     val,
                            "phi_type": m.get("label", "UNKNOWN")
                        })

        samples.append({
            "text":      text,
            "phi_spans": phi_spans
        })

    print(f"[Eval] Loaded {len(samples)} samples.")
    return samples


def load_mtsamples(n: int = EVAL_SAMPLE_SIZE) -> list[dict]:
    """
    Load MTSamples medical transcription dataset.
    Returns list of {text, phi_spans: [], dataset: 'mtsamples'} dicts.
    phi_spans is always empty — MTSamples has no PHI annotations,
    so detection metrics (P/R/F1) are not computable; latency and
    security metrics are still valid.
    """
    MTSAMPLES_NAME = "rungalileo/medical_transcription_40"
    print(f"[Eval] Loading {MTSAMPLES_NAME} ...")
    ds = load_dataset(MTSAMPLES_NAME, split="train", streaming=True)

    samples = []
    for row in ds:
        if len(samples) >= n:
            break
        text = row.get("text") or ""
        if not text.strip():
            continue
        samples.append({
            "text":      text.strip(),
            "phi_spans": [],          # no GT annotations available
            "dataset":   "mtsamples",
        })

    print(f"[Eval] Loaded {len(samples)} MTSamples samples.")
    return samples


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_detection_metrics(
    gt_spans: list[dict],
    pred_spans: list[dict]
) -> dict:
    """
    Compute precision, recall, F1 for one sample.
    Matching is exact span text match.
    """
    gt_texts   = {s["span"] for s in gt_spans}
    pred_texts = {s["span"] for s in pred_spans}

    tp = len(gt_texts & pred_texts)
    fp = len(pred_texts - gt_texts)
    fn = len(gt_texts - pred_texts)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
    }


def compute_utility_score(original: str, masked: str) -> float:
    """
    Simple utility preservation: token overlap ratio.
    High score = most non-PHI tokens preserved.
    """
    orig_tokens   = set(original.lower().split())
    masked_tokens = set(masked.lower().split())
    if not orig_tokens:
        return 1.0
    overlap = len(orig_tokens & masked_tokens) / len(orig_tokens)
    return round(overlap, 4)


# ── Runner ────────────────────────────────────────────────────────────────────

class EvalRunner:
    def __init__(self, validate: bool = True):
        self.pipeline = SELENCLLMPipeline()
        self.validate = validate
        self.records  = []   # per-sample records
        self.memory_growth = []  # (step, memory_size)

    def run(self, samples: list[dict]) -> dict:
        print(f"\n[Eval] Running on {len(samples)} samples | validate={self.validate}\n")

        per_type_metrics = defaultdict(lambda: {"tp":0,"fp":0,"fn":0})

        for i, sample in enumerate(tqdm(samples, desc="Evaluating")):
            text      = sample["text"]
            gt_spans  = sample["phi_spans"]

            try:
                result = self.pipeline.process(text, validate=self.validate)
            except Exception as e:
                print(f"[Eval] Sample {i} failed: {e}")
                continue

            pred_spans = result.get("spans", [])
            metrics    = compute_detection_metrics(gt_spans, pred_spans)
            utility    = compute_utility_score(text, result["masked"])

            # Per-type breakdown
            gt_by_type   = defaultdict(set)
            pred_by_type = defaultdict(set)
            for s in gt_spans:
                gt_by_type[s["phi_type"]].add(s["span"])
            for s in pred_spans:
                pred_by_type[s.get("phi_type", "UNKNOWN")].add(s["span"])

            for phi_type in set(list(gt_by_type.keys()) + list(pred_by_type.keys())):
                gt_t   = gt_by_type[phi_type]
                pred_t = pred_by_type[phi_type]
                per_type_metrics[phi_type]["tp"] += len(gt_t & pred_t)
                per_type_metrics[phi_type]["fp"] += len(pred_t - gt_t)
                per_type_metrics[phi_type]["fn"] += len(gt_t - pred_t)

            # Memory growth snapshot every 50 samples
            if i % 50 == 0:
                mem_stats = self.pipeline.memory.stats()
                self.memory_growth.append({
                    "step":      i,
                    "total":     mem_stats["total"],
                    "promoted":  mem_stats["promoted"],
                    "avg_conf":  mem_stats["avg_confidence"],
                })

            # Record
            record = {
                "sample_id":        i,
                "text_len":         len(text),
                "gt_span_count":    len(gt_spans),
                "pred_span_count":  len(pred_spans),
                "tp":               metrics["tp"],
                "fp":               metrics["fp"],
                "fn":               metrics["fn"],
                "precision":        metrics["precision"],
                "recall":           metrics["recall"],
                "f1":               metrics["f1"],
                "utility_score":    utility,
                "latency_a_ms":     result["latency_a_ms"],
                "latency_b_ms":     result.get("validation", {}).get("latency_b_ms", 0),
                "cache_hits":       result["cache_hits"],
                "cache_misses":     result["cache_misses"],
                "encrypted_count":  result["encrypted_count"],
                "abstracted_count": result["abstracted_count"],
                "is_clean":         result.get("validation", {}).get("is_clean", None),
                "total_reward":     result.get("validation", {}).get("total_reward", 0),
                "masked":           result["masked"],
            }
            self.records.append(record)

        return self._aggregate(per_type_metrics)

    def _aggregate(self, per_type_metrics: dict) -> dict:
        df = pd.DataFrame(self.records)

        # Overall metrics
        total_tp = df["tp"].sum()
        total_fp = df["fp"].sum()
        total_fn = df["fn"].sum()

        macro_precision = df["precision"].mean()
        macro_recall    = df["recall"].mean()
        macro_f1        = df["f1"].mean()

        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        micro_recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        micro_f1        = (
            2 * micro_precision * micro_recall / (micro_precision + micro_recall)
            if (micro_precision + micro_recall) > 0 else 0
        )

        miss_rate       = total_fn / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0

        # Latency breakdown
        cache_hit_rows  = df[df["cache_hits"] > 0]
        cache_miss_rows = df[df["cache_misses"] > 0]

        latency_stats = {
            "overall_mean_ms":   round(df["latency_a_ms"].mean(), 2),
            "overall_median_ms": round(df["latency_a_ms"].median(), 2),
            "cache_hit_mean_ms": round(cache_hit_rows["latency_a_ms"].mean(), 2)
                                  if len(cache_hit_rows) > 0 else 0,
            "cache_miss_mean_ms":round(cache_miss_rows["latency_a_ms"].mean(), 2)
                                  if len(cache_miss_rows) > 0 else 0,
            "p95_ms":            round(df["latency_a_ms"].quantile(0.95), 2),
        }

        # Per-type metrics
        per_type_rows = []
        for phi_type, counts in per_type_metrics.items():
            tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
            p  = tp / (tp + fp) if (tp + fp) > 0 else 0
            r  = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2*p*r/(p+r) if (p+r) > 0 else 0
            per_type_rows.append({
                "phi_type":  phi_type,
                "precision": round(p, 3),
                "recall":    round(r, 3),
                "f1":        round(f1, 3),
                "support":   tp + fn,
            })
        per_type_df = pd.DataFrame(per_type_rows).sort_values("support", ascending=False)

        summary = {
            "n_samples":        len(self.records),
            "macro_precision":  round(macro_precision, 4),
            "macro_recall":     round(macro_recall, 4),
            "macro_f1":         round(macro_f1, 4),
            "micro_precision":  round(micro_precision, 4),
            "micro_recall":     round(micro_recall, 4),
            "micro_f1":         round(micro_f1, 4),
            "miss_rate":        round(miss_rate, 4),
            "avg_utility_score":round(df["utility_score"].mean(), 4),
            "avg_reward":       round(df["total_reward"].mean(), 4),
            "cache_hit_rate":   round(len(cache_hit_rows) / len(df), 4),
            "latency":          latency_stats,
            "per_type":         per_type_df.to_dict(orient="records"),
            "memory_growth":    self.memory_growth,
        }

        self._save_results(df, summary, per_type_df)
        return summary

    # ── Save ─────────────────────────────────────────────────────────────────

    def _save_results(self, df: pd.DataFrame, summary: dict, per_type_df: pd.DataFrame):

        # Raw CSV
        raw_path = os.path.join(TABLES_DIR, "eval_raw.csv")
        df.to_csv(raw_path, index=False)
        print(f"[Eval] Raw results → {raw_path}")

        # Summary JSON
        summary_path = os.path.join(TABLES_DIR, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[Eval] Summary → {summary_path}")

        # Per-type CSV
        per_type_path = os.path.join(TABLES_DIR, "per_type_metrics.csv")
        per_type_df.to_csv(per_type_path, index=False)
        print(f"[Eval] Per-type → {per_type_path}")

        # Memory growth CSV
        mem_path = os.path.join(TABLES_DIR, "memory_growth.csv")
        pd.DataFrame(self.memory_growth).to_csv(mem_path, index=False)
        print(f"[Eval] Memory growth → {mem_path}")

        # LaTeX main results table
        self._save_latex_main(summary)

        # LaTeX per-type table
        self._save_latex_per_type(per_type_df)

        # Print summary to console
        self._print_summary(summary)

    def _save_latex_main(self, summary: dict):
        lat = summary["latency"]
        latex = rf"""
\begin{{table}}[h]
\centering
\caption{{SELENCLLM Edge — Overall Performance on ai4privacy ({summary['n_samples']} samples)}}
\label{{tab:main_results}}
\begin{{tabular}}{{lc}}
\hline
\textbf{{Metric}} & \textbf{{Value}} \\
\hline
Macro Precision      & {summary['macro_precision']:.4f} \\
Macro Recall         & {summary['macro_recall']:.4f} \\
Macro F1             & {summary['macro_f1']:.4f} \\
Micro Precision      & {summary['micro_precision']:.4f} \\
Micro Recall         & {summary['micro_recall']:.4f} \\
Micro F1             & {summary['micro_f1']:.4f} \\
Miss Rate            & {summary['miss_rate']:.4f} \\
Utility Score        & {summary['avg_utility_score']:.4f} \\
Avg Reward           & {summary['avg_reward']:.4f} \\
Cache Hit Rate       & {summary['cache_hit_rate']:.4f} \\
\hline
Mean Latency (ms)    & {lat['overall_mean_ms']} \\
Cache Hit Latency    & {lat['cache_hit_mean_ms']} \\
Cache Miss Latency   & {lat['cache_miss_mean_ms']} \\
P95 Latency (ms)     & {lat['p95_ms']} \\
\hline
\end{{tabular}}
\end{{table}}
"""
        path = os.path.join(TABLES_DIR, "table_main_results.tex")
        with open(path, "w") as f:
            f.write(latex)
        print(f"[Eval] LaTeX main table → {path}")

    def _save_latex_per_type(self, per_type_df: pd.DataFrame):
        rows = ""
        for _, row in per_type_df.iterrows():
            rows += (
                f"{row['phi_type']} & {row['precision']:.3f} & "
                f"{row['recall']:.3f} & {row['f1']:.3f} & {row['support']} \\\\\n"
            )

        latex = rf"""
\begin{{table}}[h]
\centering
\caption{{Per-PHI-Type Detection Performance}}
\label{{tab:per_type}}
\begin{{tabular}}{{lcccc}}
\hline
\textbf{{PHI Type}} & \textbf{{Precision}} & \textbf{{Recall}} & \textbf{{F1}} & \textbf{{Support}} \\
\hline
{rows}\hline
\end{{tabular}}
\end{{table}}
"""
        path = os.path.join(TABLES_DIR, "table_per_type.tex")
        with open(path, "w") as f:
            f.write(latex)
        print(f"[Eval] LaTeX per-type table → {path}")

    def _print_summary(self, summary: dict):
        print("\n" + "="*50)
        print("SELENCLLM EDGE — EVAL SUMMARY")
        print("="*50)
        print(f"Samples:          {summary['n_samples']}")
        print(f"Macro F1:         {summary['macro_f1']:.4f}")
        print(f"Micro F1:         {summary['micro_f1']:.4f}")
        print(f"Miss Rate:        {summary['miss_rate']:.4f}")
        print(f"Utility Score:    {summary['avg_utility_score']:.4f}")
        print(f"Cache Hit Rate:   {summary['cache_hit_rate']:.4f}")
        print(f"Avg Latency A:    {summary['latency']['overall_mean_ms']} ms")
        print(f"P95 Latency:      {summary['latency']['p95_ms']} ms")
        print(f"Avg Reward:       {summary['avg_reward']:.4f}")
        print("="*50 + "\n")


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SELENCLLM Edge Evaluator")
    parser.add_argument("--n",        type=int,  default=EVAL_SAMPLE_SIZE,
                        help="Number of samples to evaluate")
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip Agent B validation (faster, latency-only mode)")
    args = parser.parse_args()

    samples = load_ai4privacy(n=args.n)
    runner  = EvalRunner(validate=not args.no_validate)
    summary = runner.run(samples)
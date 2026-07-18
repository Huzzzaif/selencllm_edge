# SenseCLLM

> A three-agent, edge-resident framework for adaptive PII detection and selective anonymization in IoT-to-LLM pipelines.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/IEEE-AIIoT%202026-orange.svg)](#citation)

SenseCLLM detects, encrypts, and semantically abstracts personally identifiable information (PII) **entirely on the edge device** before any text is transmitted to a cloud LLM. No raw PII ever crosses the device boundary. The system improves both detection accuracy *and* inference latency over deployment — without fine-tuning, labeled data, or cloud dependency.

> **Paper:** *SenseCLLM: A Three-Agent Edge-Resident Framework for Adaptive PII Detection and Selective Anonymization in IoT-to-LLM Pipelines* — accepted at **IEEE AIIoT 2026**.

---

## Why

LLM-backed IoT pipelines routinely send raw user text to a cloud model, exposing PII to transmission interception and honest-but-curious cloud providers. Existing solutions rely on static rule sets, cloud-hosted detection, or similar-value substitution — none adapt to novel PII formats, run fully on-device, or improve with deployment. SenseCLLM closes that gap with on-device, reinforcement-driven privacy protection.

## Key results

- **F1 of 0.85** on the ai4privacy benchmark — outperforming Presidio by 49.0% and PromptObfus by 23.3%.
- **10.5× latency reduction** — mean inference drops from 1960 ms (cold start) to 187 ms after 500 samples, at a 0.78 cache hit rate.
- **Exact PII leakage rate of 0.38** and **reconstruction attack success rate of 0.23** — reductions of 52.5% and 67.6% over the strongest LLM-based baseline.
- **Utility score of 0.81** — aggressive anonymization without sacrificing downstream usability.
- Achieved **without fine-tuning, labeled data, or cloud connectivity**.

## How it works

A closed-loop, three-agent pipeline runs entirely on-device:

- **Agent A1 — PII Detector:** semantic PII detection governed by a tunable sensitivity parameter; combines regex pre-scan with on-device LLM inference.
- **Agent A2 — Pattern Learner + Masker:** queries an adaptive pattern memory for cached regex or generates new patterns via on-device LLM; routes structured PII (SSN, phone, email, date, credit card) to **ChaCha20-Poly1305** authenticated encryption and semantic PII (person, location, org, diagnosis, age) to **role-based abstraction**.
- **Agent A3 — Reinforcement Critic:** validates output via three parallel checks (leakage, fluency, over-encryption) and tunes upstream agent parameters through a reinforcement feedback signal.

The **adaptive pattern memory** starts with 19 seed patterns and expands autonomously through deployment, promoting high-confidence patterns to bypass future LLM calls.

## Architecture

```
selencllm_edge/
├── agents/        # Three-agent pipeline (A1 detector, A2 learner/masker, A3 critic)
├── core/          # Core framework / orchestration
├── encryption/    # ChaCha20-Poly1305 selective encryption + HKDF key derivation
├── memory/        # Adaptive pattern memory (persistent key-value store)
├── eval/          # Evaluation scripts and metrics
├── tests/         # Test suite
├── config.py      # Configuration
├── setup.py       # Package install
└── requirements.txt
```

**Data flow:** `Raw text → A1 (detect) → A2 (encrypt / abstract) → anonymized output → Cloud LLM`. Only the anonymized output leaves the device.

## Dataset

Evaluated on the **[ai4privacy/pii-masking-400k](https://huggingface.co/datasets/ai4privacy/pii-masking-400k)** benchmark (500-sentence English sample, natural PII distribution). The repo also includes `mtsamples.csv` (medical transcription data) for healthcare-focused testing.

## Setup

Experiments run fully on-device with no cloud connectivity.

- **Hardware:** Apple M-series SoC, 24 GB unified memory, CPU-only (also profiled on an 8 GB / 4-core config simulating an NVIDIA Jetson Orin NX gateway).
- **On-device LLM:** Llama 3.1 8B, 4-bit quantized, served locally via [Ollama](https://ollama.com).

```bash
git clone https://github.com/Huzzzaif/selencllm_edge.git
cd selencllm_edge
pip install -r requirements.txt
pip install -e .

# Pull the local model
ollama pull llama3.1:8b
```

> Requires Python 3.10+ and a running Ollama instance. _[Confirm Python version against your environment.]_

## Usage

```bash
# [Replace with your actual entry point — confirm the real command]
python -m core.run --config config.py --input mtsamples.csv
```

## Key hyperparameters

| Parameter | Value | Description |
|---|---|---|
| Seed patterns | 19 | Initial pattern memory |
| τ_apply / τ_promote | 0.75 | Confidence to apply / promote a pattern |
| τ_prune | 0.40 | Pruning confidence threshold |
| Initial sensitivity θ₁ | 0.70 | Agent A1 starting sensitivity |
| Reward weights | 1.0, 1.0, 0.5, 0.5, 0.3 | A3 critic reward terms |
| Eval / attack samples | 500 / 100 | Dataset split sizes |

## Citation

```bibtex
@inproceedings{khan2026sensecllm,
  title     = {SenseCLLM: A Three-Agent Edge-Resident Framework for Adaptive PII
               Detection and Selective Anonymization in IoT-to-LLM Pipelines},
  author    = {Khan, Huzaif and [co-authors]},
  booktitle = {IEEE AIIoT},
  year      = {2026}
}
```

> _[Fill in co-authors and the exact booktitle/full conference name from the published version.]_

## Future work

TEE-based attestation for multi-device deployments, zero-knowledge proofs for server-side anonymization verification, and domain-specific LLM fine-tuning for clinical and financial corpora.

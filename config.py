"""
SELENCLLM Edge — Central Configuration
"""

# ── Edge LLM ──────────────────────────────────────────────────────────────────
LLM_MODEL         = "llama3.1:8b"   # ollama model tag — swap to mistral etc.
LLM_TIMEOUT       = 30              # seconds per inference call

# ── Encryption ────────────────────────────────────────────────────────────────
CHACHA_KEY_SIZE   = 32              # bytes — ChaCha20-Poly1305
ENCRYPT_MARKER    = "<<ENC:"        # prefix to mark encrypted spans in output
ENCRYPT_END       = ":ENC>>"        # suffix

# ── Pattern Memory ────────────────────────────────────────────────────────────
MEMORY_PATH       = "memory/pattern_memory.json"
HIT_PROMOTE_THRESHOLD   = 20        # hits before pattern is auto-applied (skip LLM)
CONFIDENCE_PRUNE_BELOW  = 0.40      # prune patterns below this confidence
MAX_MEMORY_ENTRIES      = 200       # cap memory size

# ── RL Rewards ────────────────────────────────────────────────────────────────
REWARD_CORRECT_REDACTION  =  1.0
REWARD_MISSED_PHI         = -1.0
REWARD_OVER_ENCRYPTION    = -0.5
REWARD_CACHE_HIT          =  0.3
REWARD_UTILITY_PRESERVED  =  0.5
REWARD_REIDENTIFIABLE     = -2.0

# ── Abstraction Levels ────────────────────────────────────────────────────────
# 1 = role ("the physician"), 2 = category ("a medical professional"), 3 = abstract ("a person")
DEFAULT_ABSTRACTION_LEVEL = 1
ABSTRACTION_LEVELS = {1: "role", 2: "category", 3: "abstract"}

# ── Routing Thresholds ────────────────────────────────────────────────────────
# PHI types that always go to encrypt (need reversibility)
ENCRYPT_TYPES = {"SSN", "DOB", "PHONE", "EMAIL", "ID_NUMBER", "CREDIT_CARD", "DATE"}
# PHI types that go to semantic abstraction
ABSTRACT_TYPES = {"PERSON", "LOCATION", "ORG", "DIAGNOSIS", "AGE", "DATE", "RELATIONSHIP"}

# ── Agent Parameter Bounds (Agent 3 enforces these) ──────────────────────────
SENSITIVITY_DEFAULT        = 0.7
SENSITIVITY_MIN            = 0.3
SENSITIVITY_MAX            = 1.0
SENSITIVITY_STEP           = 0.05

MIN_CONFIDENCE_DEFAULT     = 0.75
MIN_CONFIDENCE_MIN         = 0.50
MIN_CONFIDENCE_MAX         = 0.95

PROMOTE_THRESHOLD_DEFAULT  = 20
PROMOTE_THRESHOLD_MIN      = 5
PROMOTE_THRESHOLD_MAX      = 50

ABSTRACTION_LEVEL_DEFAULT  = 1     # 1=role, 2=category, 3=abstract

# ── Eval ──────────────────────────────────────────────────────────────────────
DATASET_NAME      = "ai4privacy/pii-masking-400k"
EVAL_SAMPLE_SIZE  = 500             # rows to eval on
"""
memory/seed_patterns.py
Pre-built high-confidence PHI patterns to bootstrap the pattern memory.
These skip the LLM detection step on first run (promoted=True equivalent via high hits).
"""

from memory.pattern_memory import PatternMemory

# (label, regex, phi_type, abstraction_role)
SEED_PATTERNS = [
    # SSN
    ("SSN_standard",        r"\b\d{3}-\d{2}-\d{4}\b",                                        "SSN",      None),

    # Phone
    ("PHONE_us_dashes",     r"\b\d{3}-\d{3}-\d{4}\b",                                        "PHONE",    None),
    ("PHONE_us_dots",       r"\b\d{3}\.\d{3}\.\d{4}\b",                                      "PHONE",    None),
    ("PHONE_us_parens",     r"\(\d{3}\)\s*\d{3}-\d{4}",                                      "PHONE",    None),

    # Email
    ("EMAIL_standard",      r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",        "EMAIL",    None),

    # Date of birth / dates
    ("DATE_mdy_slash",      r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",                                  "DATE",     None),
    ("DATE_mdy_dash",       r"\b\d{1,2}-\d{1,2}-\d{2,4}\b",                                  "DATE",     None),
    ("DATE_month_name",     r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
                            r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|"
                            r"Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b",
                                                                                               "DATE",     None),

    # Age
    ("AGE_year_old",        r"\b\d{1,3}[-\s]year[-\s]old\b",                                  "AGE",      "a middle-aged adult"),
    ("AGE_years_of_age",    r"\b\d{1,3}\s+years?\s+of\s+age\b",                               "AGE",      "a middle-aged adult"),

    # ZIP code
    ("LOCATION_zip",        r"\b\d{5}(?:-\d{4})?\b",                                          "LOCATION", "a local area"),

    # Credit card
    ("CREDIT_CARD_visa",    r"\b4\d{3}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b",              "CREDIT_CARD", None),
    ("CREDIT_CARD_generic", r"\b\d{4}[\s\-]\d{4}[\s\-]\d{4}[\s\-]\d{4}\b",                   "CREDIT_CARD", None),

    # ID numbers
    ("ID_NUMBER_mrn",       r"\bMRN\s*[:#]?\s*\d{5,10}\b",                                    "ID_NUMBER", None),
    ("ID_NUMBER_patient",   r"\bPatient\s+ID\s*[:#]?\s*\d{4,10}\b",                           "ID_NUMBER", None),

    # Relationships
    ("RELATIONSHIP_spouse", r"\b(?:wife|husband|spouse|partner)\b",                            "RELATIONSHIP", "a family member"),
    ("RELATIONSHIP_child",  r"\b(?:son|daughter|child|children)\b",                            "RELATIONSHIP", "a family member"),
]


def seed_memory(memory: PatternMemory):
    """
    Add seed patterns to memory if they don't already exist.
    Sets high confidence + promoted so they fire without LLM on first run.
    """
    for label, regex, phi_type, abstraction_role in SEED_PATTERNS:
        if label in memory.memory:
            continue
        memory.memory[label] = {
            "regex":            regex,
            "phi_type":         phi_type,
            "abstraction_role": abstraction_role,
            "hits":             48,          # confidence 0.96 — above any reasonable min_confidence_to_apply
            "misses":           2,
            "confidence":       round(48 / 50, 3),
            "promoted":         True,
            "last_updated":     "2026-04-02",
        }
    memory._save()


if __name__ == "__main__":
    mem = PatternMemory()
    seed_memory(mem)
    s = mem.stats()
    print(f"Seeded. Total={s['total']} Promoted={s['promoted']} AvgConf={s['avg_confidence']}")

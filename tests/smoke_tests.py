"""
tests/smoke_test.py
Quick end-to-end sanity check — run this before the full eval.

Usage:
    python tests/smoke_test.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pipeline import SELENCLLMPipeline
from encryption.chacha import ChaChaEncryptor
from memory.pattern_memory import PatternMemory


TEST_SENTENCES = [
    "John Smith was diagnosed with HIV and lives at 123 Main St, Compton, CA.",
    "Patient SSN is 123-45-6789 and DOB is 03/14/1985.",
    "Dr. Ahmed prescribed metformin to the 42-year-old male.",
    "Call Mary Johnson at 555-867-5309 or email mary@example.com.",
    "The patient was admitted to Cedars-Sinai on March 3rd, 2024.",
]


def test_chacha():
    print("\n── Test 1: ChaCha20 Encryption ──")
    enc = ChaChaEncryptor()
    token, record = enc.encrypt_span("123-45-6789")
    decrypted     = enc.decrypt_span(token, record)
    assert decrypted == "123-45-6789", f"Decrypt failed: {decrypted}"

    masked, vault = enc.encrypt_sentence(
        "SSN is 123-45-6789 and phone is 555-1234",
        ["123-45-6789", "555-1234"]
    )
    restored = enc.decrypt_sentence(masked, vault)
    assert "123-45-6789" in restored
    assert "555-1234"    in restored
    print("  ✅ ChaCha20 encrypt/decrypt OK")


def test_pattern_memory():
    print("\n── Test 2: Pattern Memory ──")
    tmp_path = "/tmp/test_memory.json"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    mem = PatternMemory(path=tmp_path)
    mem.add(
        label="SSN_test",
        regex=r"\b\d{3}-\d{2}-\d{4}\b",
        phi_type="SSN",
        abstraction_role=None
    )
    matches = mem.match("His SSN is 123-45-6789.")
    assert len(matches) == 1
    assert matches[0]["span"] == "123-45-6789"

    mem.update_hit("SSN_test", correct=True)
    assert mem.memory["SSN_test"]["hits"] == 1
    print("  ✅ Pattern memory add/match/update OK")


def test_pipeline_no_validate():
    print("\n── Test 3: Pipeline (Agent A only, no validation) ──")
    pipeline = SELENCLLMPipeline()

    for sentence in TEST_SENTENCES:
        print(f"\n  Input:  {sentence}")
        result = pipeline.process(sentence, validate=False)
        print(f"  Masked: {result['masked']}")
        print(f"  Spans:  {[(s['span'], s['action']) for s in result['spans']]}")
        print(f"  Latency A: {result['latency_a_ms']} ms")
        print(f"  Cache hits: {result['cache_hits']} | misses: {result['cache_misses']}")

    print("\n  ✅ Pipeline (no validate) OK")


def test_pipeline_with_validate():
    print("\n── Test 4: Full Pipeline (Agent A + B) ──")
    pipeline = SELENCLLMPipeline()
    sentence = TEST_SENTENCES[0]

    print(f"\n  Input:  {sentence}")
    result = pipeline.process(sentence, validate=True)
    print(f"  Masked: {result['masked']}")

    val = result.get("validation", {})
    print(f"  Is clean:     {val.get('is_clean')}")
    print(f"  Is fluent:    {val.get('is_fluent')}")
    print(f"  Fluency score:{val.get('fluency_score')}")
    print(f"  Total reward: {val.get('total_reward')}")
    print(f"  Latency A: {result['latency_a_ms']} ms | B: {val.get('latency_b_ms')} ms")

    print("\n  ✅ Full pipeline OK")


def test_decrypt():
    print("\n── Test 5: Decrypt Encrypted Spans ──")
    pipeline = SELENCLLMPipeline()
    sentence = "Patient SSN is 123-45-6789."
    result   = pipeline.process(sentence, validate=False)

    if result["vault_snapshot"]:
        restored = pipeline.decrypt(result["masked"], result["vault_snapshot"])
        assert "123-45-6789" in restored, f"Decrypt failed: {restored}"
        print(f"  Masked:   {result['masked']}")
        print(f"  Restored: {restored}")
        print("  ✅ Decrypt OK")
    else:
        print("  ⚠️  No encrypted spans in output — SSN may have been abstracted")


if __name__ == "__main__":
    print("=" * 50)
    print("SELENCLLM Edge — Smoke Test")
    print("=" * 50)

    try:
        test_chacha()
        test_pattern_memory()
        test_pipeline_no_validate()
        test_pipeline_with_validate()
        test_decrypt()
        print("\n" + "="*50)
        print("ALL TESTS PASSED ✅")
        print("="*50)
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        raise
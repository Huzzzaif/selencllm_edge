"""
encryption/chacha.py
Selective ChaCha20-Poly1305 encryption for sensitive spans.

Usage:
    enc = ChaChaEncryptor()
    token, record = enc.encrypt_span("123-45-6789")
    original      = enc.decrypt_span(token, record)
"""

import os
import base64
import json
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from config import CHACHA_KEY_SIZE, ENCRYPT_MARKER, ENCRYPT_END


class ChaChaEncryptor:
    def __init__(self, key: bytes = None):
        """
        key — 32-byte ChaCha20 key.
             If None, a fresh key is generated per session.
             For persistence across calls, pass in a saved key.
        """
        self.key   = key or os.urandom(CHACHA_KEY_SIZE)
        self.cipher = ChaCha20Poly1305(self.key)
        # In-session vault: token → {nonce, ciphertext}
        self._vault: dict[str, dict] = {}

    # ── Core ──────────────────────────────────────────────────────────────────

    def encrypt_span(self, plaintext: str) -> tuple[str, dict]:
        """
        Encrypt a sensitive span.
        Returns:
            token   — inline placeholder e.g. <<ENC:abc123:ENC>>
            record  — {nonce_b64, ciphertext_b64} stored in vault
        """
        nonce      = os.urandom(12)                          # 96-bit nonce
        ciphertext = self.cipher.encrypt(nonce, plaintext.encode(), None)

        nonce_b64  = base64.b64encode(nonce).decode()
        ct_b64     = base64.b64encode(ciphertext).decode()

        token_id   = base64.urlsafe_b64encode(os.urandom(8)).decode().rstrip("=")
        token      = f"{ENCRYPT_MARKER}{token_id}{ENCRYPT_END}"

        record = {"nonce": nonce_b64, "ciphertext": ct_b64, "original_len": len(plaintext)}
        self._vault[token_id] = record

        return token, record

    def decrypt_span(self, token: str, record: dict = None) -> str:
        """
        Decrypt a span given its token.
        record is optional if token was already stored in vault.
        """
        token_id = token.replace(ENCRYPT_MARKER, "").replace(ENCRYPT_END, "")
        rec      = record or self._vault.get(token_id)

        if rec is None:
            raise KeyError(f"No vault record for token: {token_id}")

        nonce      = base64.b64decode(rec["nonce"])
        ciphertext = base64.b64decode(rec["ciphertext"])
        plaintext  = self.cipher.decrypt(nonce, ciphertext, None)

        return plaintext.decode()

    def encrypt_sentence(self, sentence: str, spans: list[str]) -> tuple[str, dict]:
        """
        Given a sentence and a list of sensitive spans (strings),
        replace each span with its encrypted token in-place.
        Returns:
            masked_sentence — sentence with tokens replacing spans
            vault_snapshot  — {token_id: record} for all spans in this sentence
        """
        masked    = sentence
        snapshot  = {}

        for span in spans:
            if span not in masked:
                continue
            token, record = self.encrypt_span(span)
            token_id = token.replace(ENCRYPT_MARKER, "").replace(ENCRYPT_END, "")
            masked   = masked.replace(span, token, 1)
            snapshot[token_id] = record

        return masked, snapshot

    def decrypt_sentence(self, masked_sentence: str, vault_snapshot: dict) -> str:
        """Restore all encrypted tokens in a sentence back to plaintext."""
        result = masked_sentence
        for token_id, record in vault_snapshot.items():
            token    = f"{ENCRYPT_MARKER}{token_id}{ENCRYPT_END}"
            if token in result:
                plaintext = self.decrypt_span(token, record)
                result    = result.replace(token, plaintext)
        return result

    # ── Key management ────────────────────────────────────────────────────────

    def export_key(self) -> str:
        return base64.b64encode(self.key).decode()

    @classmethod
    def from_key_b64(cls, key_b64: str) -> "ChaChaEncryptor":
        return cls(key=base64.b64decode(key_b64))

    def export_vault(self) -> str:
        return json.dumps(self._vault)

    def import_vault(self, vault_json: str):
        self._vault.update(json.loads(vault_json))
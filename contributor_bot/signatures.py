"""Webhook and interaction signature checks.

GitHub signs webhook bodies with HMAC-SHA256. Discord signs interaction
bodies with Ed25519 over ``timestamp || body``. Verification is pure
Python so the bot stays dependency-free. Signing is intentionally not
implemented: the bot only needs to check signatures it receives.
"""

import hashlib
import hmac
import logging
import re
import time

_P = 2**255 - 19
_L = 2**252 + 27742317777372353535851937790883648493
_D = (-121665 * pow(121666, _P - 2, _P)) % _P
_I = pow(2, (_P - 1) // 4, _P)
_TIMESTAMP = re.compile(r"[0-9]{1,12}")
# Discord recommends rejecting interaction timestamps outside a few minutes.
MAX_INTERACTION_SKEW_SECONDS = 5 * 60


def verify_github_signature(secret: str, body: bytes, header: str | None) -> bool:
    """Return True when ``header`` is ``sha256=<hex>`` for ``body``."""
    if not secret or not header or not header.startswith("sha256="):
        return False
    provided = header[len("sha256="):]
    if not re.fullmatch(r"[0-9a-f]{64}", provided.lower()):
        return False
    expected = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected.encode("ascii"), provided.lower().encode("ascii"))


def _recover_x(y: int, sign: int) -> int | None:
    if y >= _P:
        return None
    y2 = y * y % _P
    denom = (_D * y2 + 1) % _P
    if denom == 0:
        return None
    x2 = (y2 - 1) * pow(denom, _P - 2, _P) % _P
    if x2 == 0:
        if sign:
            return None
        return 0
    x = pow(x2, (_P + 3) // 8, _P)
    if (x * x - x2) % _P != 0:
        x = x * _I % _P
    if (x * x - x2) % _P != 0:
        return None
    if (x & 1) != sign:
        x = (-x) % _P
    return x


def _decode_point(data: bytes) -> tuple[int, int] | None:
    if len(data) != 32:
        return None
    raw = int.from_bytes(data, "little")
    sign = (raw >> 255) & 1
    y = raw & ((1 << 255) - 1)
    x = _recover_x(y, sign)
    if x is None:
        return None
    if (-x * x + y * y - 1 - _D * x * x * y * y) % _P != 0:
        return None
    return x, y


def _edwards_add(p: tuple[int, int], q: tuple[int, int]) -> tuple[int, int]:
    x1, y1 = p
    x2, y2 = q
    product = _D * x1 * x2 * y1 * y2
    x3 = (x1 * y2 + y1 * x2) * pow(1 + product, _P - 2, _P) % _P
    y3 = (y1 * y2 + x1 * x2) * pow(1 - product, _P - 2, _P) % _P
    return x3, y3


def _scalarmult(point: tuple[int, int], scalar: int) -> tuple[int, int]:
    result = (0, 1)
    addend = point
    value = scalar
    while value:
        if value & 1:
            result = _edwards_add(result, addend)
        addend = _edwards_add(addend, addend)
        value >>= 1
    return result


def _base_point() -> tuple[int, int]:
    # RFC 8032 base point. x is even.
    y = 4 * pow(5, _P - 2, _P) % _P
    x = _recover_x(y, 0)
    if x is None:
        raise RuntimeError("Ed25519 base point is not on the curve")
    return x, y


_BASE = _base_point()


def ed25519_verify(public_key: bytes, message: bytes, signature: bytes) -> bool:
    """RFC 8032 Ed25519 verification (cofactorless, S and k reduced mod L)."""
    if len(public_key) != 32 or len(signature) != 64:
        return False
    try:
        point_r = _decode_point(signature[:32])
        point_a = _decode_point(public_key)
        if point_r is None or point_a is None:
            return False
        scalar_s = int.from_bytes(signature[32:], "little")
        if scalar_s >= _L:
            return False
        digest = hashlib.sha512(signature[:32] + public_key + message).digest()
        scalar_k = int.from_bytes(digest, "little") % _L
        return _scalarmult(_BASE, scalar_s) == _edwards_add(point_r, _scalarmult(point_a, scalar_k))
    except Exception:
        logging.exception("Ed25519 verification failed closed")
        return False


def ed25519_public_key_ok(public_key_hex: str) -> bool:
    try:
        raw = bytes.fromhex(public_key_hex)
    except ValueError:
        return False
    return _decode_point(raw) is not None


def verify_discord_interaction(
    public_key_hex: str,
    signature_hex: str | None,
    timestamp: str | None,
    body: bytes,
    *,
    now: float | None = None,
) -> bool:
    """Check a Discord interaction request.

    The signed message is the timestamp header concatenated with the raw body.
    Timestamps more than five minutes from ``now`` are rejected. The timestamp
    is part of the signed message, so it cannot be swapped onto another body.
    """
    if not timestamp or not _TIMESTAMP.fullmatch(timestamp) or not signature_hex:
        return False
    skew = abs((now if now is not None else time.time()) - int(timestamp))
    if skew > MAX_INTERACTION_SKEW_SECONDS:
        return False
    try:
        public_key = bytes.fromhex(public_key_hex)
        signature = bytes.fromhex(signature_hex)
    except ValueError:
        return False
    return ed25519_verify(public_key, timestamp.encode("ascii") + body, signature)

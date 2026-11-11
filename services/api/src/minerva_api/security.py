"""Password hashing, session tokens and a small in-process rate limiter."""

from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
import time
from collections import defaultdict, deque

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

_hasher = PasswordHasher()  # Argon2id with the library's current recommended parameters
_DUMMY_HASH = _hasher.hash("minerva-dummy-password")


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(stored_hash: str | None, password: str) -> bool:
    """Constant-work verification: always performs one Argon2 verification."""
    try:
        return _hasher.verify(stored_hash or _DUMMY_HASH, password) and stored_hash is not None
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def needs_rehash(stored_hash: str) -> bool:
    return _hasher.check_needs_rehash(stored_hash)


def new_session_token() -> str:
    return secrets.token_urlsafe(32)


def token_digest(token: str, secret: str) -> str:
    """HMAC of the token; only this digest is stored, never the token itself."""
    return hmac.new(secret.encode(), token.encode(), hashlib.sha256).hexdigest()


class RateLimiter:
    """Sliding-window limiter, per process. A shared limiter (Cloudflare, Redis) is a later step."""

    def __init__(self) -> None:
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str, limit: int, window_seconds: float = 60.0) -> bool:
        now = time.monotonic()
        with self._lock:
            events = self._events[key]
            while events and now - events[0] > window_seconds:
                events.popleft()
            if len(events) >= limit:
                return False
            events.append(now)
            if len(self._events) > 10_000:  # bound memory
                for k in [k for k, v in self._events.items() if not v][:1000]:
                    del self._events[k]
            return True

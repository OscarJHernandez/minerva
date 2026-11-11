import time

from fastapi.testclient import TestClient
from minerva_api.security import RateLimiter

from .conftest import PASSWORD, build_app


def test_rate_limiter_blocks_then_recovers() -> None:
    rl = RateLimiter()
    assert all(rl.allow("k", 3, 0.2) for _ in range(3))
    assert not rl.allow("k", 3, 0.2)
    time.sleep(0.25)
    assert rl.allow("k", 3, 0.2)


def test_rate_limiter_evicts_idle_keys() -> None:
    rl = RateLimiter()
    for i in range(10_050):
        rl.allow(f"ip-{i}", 5, 0.05)
    time.sleep(0.1)
    rl.allow("trigger", 5, 0.05)  # over the size bound: idle keys are dropped
    assert len(rl._events) < 100


def test_forwarded_header_is_ignored_unless_configured() -> None:
    app = build_app(auth_rate_per_minute=3)
    with TestClient(app) as c:
        codes = [
            c.post(
                "/v1/auth/login",
                json={"email": "a@example.com", "password": "wrong-password-1"},
                headers={"X-Forwarded-For": f"10.0.0.{i}", "CF-Connecting-IP": f"10.1.0.{i}"},
            ).status_code
            for i in range(6)
        ]
    assert 429 in codes  # spoofed headers do not create fresh buckets


def test_trusted_header_separates_clients_when_configured() -> None:
    app = build_app(auth_rate_per_minute=2, trusted_ip_header="cf-connecting-ip")
    with TestClient(app) as c:

        def attempt(ip: str) -> int:
            return c.post(
                "/v1/auth/login",
                json={"email": "a@example.com", "password": "wrong-password-1"},
                headers={"CF-Connecting-IP": ip},
            ).status_code

        assert [attempt("1.1.1.1") for _ in range(3)][-1] == 429
        assert attempt("2.2.2.2") == 401  # a different client is not throttled


def test_session_lookup_does_not_hold_a_transaction_open(app) -> None:  # type: ignore[no-untyped-def]
    from minerva_api.deps import DbDep, UserDep

    @app.get("/_probe")
    def probe(user: UserDep, db: DbDep) -> dict[str, bool]:
        return {"in_transaction": db.in_transaction()}

    with TestClient(app) as c:
        c.post("/v1/auth/register", json={"email": "t@example.com", "password": PASSWORD})
        # After authentication the pooled connection is released, so a slow computation
        # that follows does not pin it.
        assert c.get("/_probe").json() == {"in_transaction": False}

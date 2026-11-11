from fastapi.testclient import TestClient

from .conftest import PASSWORD, build_app, register


def test_me_requires_login(client: TestClient) -> None:
    assert client.get("/v1/auth/me").status_code == 401


def test_register_login_logout_flow(client: TestClient) -> None:
    register(client)
    assert client.get("/v1/auth/me").json()["email"] == "ada@example.com"
    assert client.post("/v1/auth/logout").status_code == 204
    assert client.get("/v1/auth/me").status_code == 401
    r = client.post("/v1/auth/login", json={"email": "ADA@example.com", "password": PASSWORD})
    assert r.status_code == 200
    assert client.get("/v1/auth/me").status_code == 200


def test_cookie_is_httponly_and_samesite(client: TestClient) -> None:
    r = client.post("/v1/auth/register", json={"email": "b@example.com", "password": PASSWORD})
    cookie = r.headers["set-cookie"].lower()
    assert "httponly" in cookie
    assert "samesite=lax" in cookie
    assert PASSWORD not in r.text


def test_duplicate_email_rejected(client: TestClient) -> None:
    register(client)
    r = client.post("/v1/auth/register", json={"email": "ada@example.com", "password": PASSWORD})
    assert r.status_code == 409


def test_weak_or_invalid_credentials_rejected(client: TestClient) -> None:
    assert (
        client.post(
            "/v1/auth/register", json={"email": "x@example.com", "password": "short"}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/v1/auth/register", json={"email": "not-an-email", "password": PASSWORD}
        ).status_code
        == 422
    )


def test_wrong_password_and_unknown_user_look_identical(client: TestClient) -> None:
    register(client)
    client.post("/v1/auth/logout")
    wrong = client.post(
        "/v1/auth/login", json={"email": "ada@example.com", "password": "wrong-password-1"}
    )
    unknown = client.post(
        "/v1/auth/login", json={"email": "nobody@example.com", "password": "wrong-password-1"}
    )
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json()


def test_forged_cookie_rejected(client: TestClient) -> None:
    client.cookies.set("minerva_session", "not-a-real-token")
    assert client.get("/v1/auth/me").status_code == 401


def test_login_rate_limit() -> None:
    app = build_app(auth_rate_per_minute=3)
    with TestClient(app) as c:
        codes = [
            c.post(
                "/v1/auth/login", json={"email": "a@example.com", "password": "wrong-password-1"}
            ).status_code
            for _ in range(5)
        ]
    assert codes[:3] == [401, 401, 401]
    assert 429 in codes[3:]


def test_registration_can_be_disabled() -> None:
    app = build_app(allow_registration=False)
    with TestClient(app) as c:
        r = c.post("/v1/auth/register", json={"email": "a@example.com", "password": PASSWORD})
        assert r.status_code == 403


def test_cross_origin_post_rejected(client: TestClient) -> None:
    r = client.post(
        "/v1/auth/register",
        json={"email": "evil@example.com", "password": PASSWORD},
        headers={"Origin": "https://evil.example"},
    )
    assert r.status_code == 403
    ok = client.post(
        "/v1/auth/register",
        json={"email": "good@example.com", "password": PASSWORD},
        headers={"Origin": "http://localhost:3000"},
    )
    assert ok.status_code == 201


def test_production_refuses_placeholder_secret() -> None:
    import pytest
    from minerva_api.settings import Settings
    from pydantic import SecretStr

    placeholder = SecretStr("change-me-generate-a-random-value")
    with pytest.raises(RuntimeError):
        Settings(
            _env_file=None, env="production", cookie_secure=True, session_secret=placeholder
        ).validate_for_runtime()
    with pytest.raises(RuntimeError):  # strong secret but insecure cookies
        Settings(
            _env_file=None,
            env="production",
            cookie_secure=False,
            session_secret=SecretStr("x" * 40),
        ).validate_for_runtime()
    Settings(
        _env_file=None, env="production", cookie_secure=True, session_secret=SecretStr("x" * 40)
    ).validate_for_runtime()  # a proper production configuration is accepted

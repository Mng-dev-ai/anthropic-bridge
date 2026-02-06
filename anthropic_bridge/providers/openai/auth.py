import base64
import json
import time
from pathlib import Path
from typing import Any, cast

import aiofiles
import httpx

TOKEN_URL = "https://auth.openai.com/oauth/token"
CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
AUTH_FILE_PATH = Path.home() / ".codex" / "auth.json"


def auth_file_exists() -> bool:
    return AUTH_FILE_PATH.exists()


async def read_auth_file() -> dict[str, Any]:
    if not AUTH_FILE_PATH.exists():
        raise RuntimeError(
            f"Auth file not found at {AUTH_FILE_PATH}. Run 'codex login' first."
        )
    async with aiofiles.open(AUTH_FILE_PATH, "r") as f:
        content = await f.read()
    return cast(dict[str, Any], json.loads(content))


def parse_jwt_expiry(token: str) -> float:
    """Extract expiry time from JWT token."""
    try:
        payload = token.split(".")[1]
        padding = 4 - len(payload) % 4
        payload += "=" * padding
        claims = json.loads(base64.urlsafe_b64decode(payload))
        return float(claims.get("exp", 0))
    except Exception:
        return 0


def extract_account_id(tokens: dict[str, Any]) -> str | None:
    """Extract ChatGPT account ID from tokens."""
    for token_key in ("id_token", "access_token"):
        token = tokens.get(token_key)
        if not token:
            continue
        try:
            payload = token.split(".")[1]
            padding = 4 - len(payload) % 4
            payload += "=" * padding
            claims = json.loads(base64.urlsafe_b64decode(payload))

            account_id: str | None = (
                claims.get("chatgpt_account_id")
                or claims.get("https://api.openai.com/auth", {}).get(
                    "chatgpt_account_id"
                )
                or (
                    claims.get("organizations", [{}])[0].get("id")
                    if claims.get("organizations")
                    else None
                )
            )
            if account_id:
                return account_id
        except Exception:
            continue
    return cast(str | None, tokens.get("account_id"))


async def refresh_tokens(refresh_token: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": CODEX_CLIENT_ID,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        if response.status_code != 200:
            raise RuntimeError(f"Token refresh failed: {response.text}")

        return cast(dict[str, Any], response.json())


async def get_auth(
    cached_token: str | None,
    cached_account_id: str | None,
    expires_at: float,
) -> tuple[str, str | None, float]:
    """Get access token and account ID for Codex API.

    Returns: (access_token, account_id, expires_at)
    """
    if cached_token and time.time() < expires_at:
        return cached_token, cached_account_id, expires_at

    auth_data = await read_auth_file()
    tokens = auth_data.get("tokens", {})
    access_token = tokens.get("access_token")
    refresh_token = tokens.get("refresh_token")

    if not access_token:
        raise RuntimeError("No access_token in auth file. Run 'codex login' first.")

    token_expiry = parse_jwt_expiry(access_token)
    if token_expiry and time.time() < token_expiry - 60:
        account_id = extract_account_id(tokens)
        return access_token, account_id, token_expiry - 60

    if not refresh_token:
        raise RuntimeError("Token expired and no refresh_token. Run 'codex login'.")

    new_tokens = await refresh_tokens(refresh_token)

    auth_data.setdefault("tokens", {}).update(new_tokens)
    try:
        async with aiofiles.open(AUTH_FILE_PATH, "w") as f:
            await f.write(json.dumps(auth_data, indent=2))
    except PermissionError:
        pass

    new_access_token = new_tokens.get("access_token", access_token)
    new_expiry = parse_jwt_expiry(new_access_token)
    if not new_expiry:
        new_expiry = time.time() + 3600

    account_id = extract_account_id(new_tokens) or extract_account_id(tokens)
    return new_access_token, account_id, new_expiry - 60

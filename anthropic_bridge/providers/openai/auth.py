import json
import logging
import time
from pathlib import Path
from typing import Any, cast

import httpx

logger = logging.getLogger(__name__)

TOKEN_EXCHANGE_URL = "https://auth.openai.com/oauth/token"
CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
AUTH_FILE_PATH = Path.home() / ".codex" / "auth.json"


def auth_file_exists() -> bool:
    return AUTH_FILE_PATH.exists()


async def read_auth_file() -> dict[str, Any]:
    if not AUTH_FILE_PATH.exists():
        raise RuntimeError(
            f"Auth file not found at {AUTH_FILE_PATH}. Run 'openai login' first."
        )
    return cast(dict[str, Any], json.loads(AUTH_FILE_PATH.read_text()))


async def exchange_token(id_token: str) -> tuple[str, float]:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            TOKEN_EXCHANGE_URL,
            data={
                "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
                "requested_token_type": "openai-api-key",
                "subject_token": id_token,
                "subject_token_type": "urn:ietf:params:oauth:token-type:id_token",
                "client_id": CODEX_CLIENT_ID,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        if response.status_code != 200:
            raise RuntimeError(f"Token exchange failed: {response.text}")

        data = response.json()
        api_key = data.get("access_token")
        expires_in = data.get("expires_in", 3600)

        if not api_key:
            raise RuntimeError("No access_token in token exchange response")

        expires_at = time.time() + expires_in - 60
        return api_key, expires_at


async def refresh_tokens(refresh_token: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            TOKEN_EXCHANGE_URL,
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


async def get_api_key(
    cached_key: str | None, expires_at: float
) -> tuple[str, float]:
    if cached_key and time.time() < expires_at:
        return cached_key, expires_at

    auth_data = await read_auth_file()
    id_token = auth_data.get("id_token")
    refresh_token = auth_data.get("refresh_token")

    if not id_token:
        raise RuntimeError("No id_token in auth file. Run 'openai login' first.")

    try:
        api_key, new_expires_at = await exchange_token(id_token)
    except RuntimeError:
        if refresh_token:
            logger.info("id_token expired, refreshing tokens...")
            new_tokens = await refresh_tokens(refresh_token)

            auth_data.update(new_tokens)
            AUTH_FILE_PATH.write_text(json.dumps(auth_data, indent=2))

            new_id_token = new_tokens.get("id_token", id_token)
            api_key, new_expires_at = await exchange_token(new_id_token)
        else:
            raise

    return api_key, new_expires_at

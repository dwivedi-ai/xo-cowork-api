"""
OpenAI Codex device-code OAuth setup.

Flow:
  1. POST /codex/setup  -> SSE stream
       a. Calls OpenAI deviceauth/usercode  -> gets device_auth_id + user_code
       b. Emits {type:"user_code", code:"XXXX-XXXX", url:"https://auth.openai.com/codex/device"}
       c. Polls deviceauth/token until user completes login
       d. Exchanges authorization_code for tokens (PKCE values come from OpenAI)
       e. Persists credentials, emits {type:"done"}

No redirect URI or local server required — works in any remote environment.
Requires the user to have "Device Code" enabled in ChatGPT security settings.

On success, persists credentials to:
  - ~/.openclaw/agents/main/agent/auth-profiles.json  (tokens / secrets)
  - ~/.openclaw/openclaw.json                          (metadata)
  - Project & OpenClaw .env files                      (OPENAI_CODEX_ACCESS_TOKEN)
"""

import asyncio
import base64
import json
import os
import time
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Optional

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse


# =============================================================================
# OAuth constants
# =============================================================================

OPENAI_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"

# Device code endpoints (from codex-rs/login/src/device_code_auth.rs)
DEVICE_USERCODE_URL = "https://auth.openai.com/api/accounts/deviceauth/usercode"
DEVICE_TOKEN_URL = "https://auth.openai.com/api/accounts/deviceauth/token"
DEVICE_VERIFICATION_URL = "https://auth.openai.com/codex/device"
DEVICE_CALLBACK_REDIRECT_URI = "https://auth.openai.com/deviceauth/callback"

# Standard PKCE token exchange
OPENAI_TOKEN_URL = "https://auth.openai.com/oauth/token"

# Credential file constants
PROVIDER_ID = "openai-codex"
AUTH_STORE_VERSION = 1

CODEX_SETUP_TIMEOUT_SECONDS = int(os.getenv("CODEX_SETUP_TIMEOUT", "900"))  # 15 min matches CLI
MIN_POLL_INTERVAL = 5  # seconds


# =============================================================================
# JWT identity extraction
# =============================================================================

def _decode_jwt_payload(token: str) -> Dict[str, Any]:
    parts = token.split(".")
    if len(parts) < 2:
        return {}
    payload_b64 = parts[1]
    padding = 4 - len(payload_b64) % 4
    if padding != 4:
        payload_b64 += "=" * padding
    try:
        return json.loads(base64.urlsafe_b64decode(payload_b64))
    except Exception:
        return {}


def resolve_codex_auth_identity(
    access_token: str,
    email_hint: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    claims = _decode_jwt_payload(access_token)

    profile_claim = claims.get("https://api.openai.com/profile", {})
    email = profile_claim.get("email") or email_hint

    auth_claim = claims.get("https://api.openai.com/auth", {})
    profile_name: Optional[str] = email

    if not profile_name and auth_claim.get("chatgpt_account_user_id"):
        raw = auth_claim["chatgpt_account_user_id"].encode("utf-8")
        profile_name = base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")
    if not profile_name:
        profile_name = auth_claim.get("chatgpt_user_id") or auth_claim.get("user_id")
    if not profile_name and claims.get("iss") and claims.get("sub"):
        profile_name = f"{claims['iss']}|{claims['sub']}"

    account_id = auth_claim.get("chatgpt_account_id")

    return {
        "email": email,
        "profile_name": profile_name or "default",
        "account_id": account_id,
    }


# =============================================================================
# Credential file I/O
# =============================================================================

def _resolve_state_dir() -> str:
    return os.environ.get("OPENCLAW_STATE_DIR") or str(Path.home() / ".openclaw")


def _resolve_auth_store_path() -> str:
    return str(Path(_resolve_state_dir()) / "agents" / "main" / "agent" / "auth-profiles.json")


def _resolve_config_path() -> str:
    return str(Path(_resolve_state_dir()) / "openclaw.json")


def _read_json(path: str, fallback: Any) -> Any:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return fallback


def _write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def write_auth_credentials(
    profile_name: str,
    access: str,
    refresh: str,
    expires: int,
    email: Optional[str] = None,
) -> None:
    auth_path = _resolve_auth_store_path()
    config_path = _resolve_config_path()
    profile_id = f"{PROVIDER_ID}:{profile_name}"

    store = _read_json(auth_path, {"version": AUTH_STORE_VERSION, "profiles": {}})
    store["version"] = AUTH_STORE_VERSION
    store.setdefault("profiles", {})[profile_id] = {
        "type": "oauth",
        "provider": PROVIDER_ID,
        "access": access,
        "refresh": refresh,
        "expires": expires,
        "email": email,
    }
    order = store.setdefault("order", {})
    provider_order = order.setdefault(PROVIDER_ID, [])
    if profile_name not in provider_order:
        provider_order.append(profile_name)
    store.setdefault("lastGood", {})[PROVIDER_ID] = profile_name
    _write_json(auth_path, store)

    config = _read_json(config_path, {})
    auth_section = config.setdefault("auth", {})
    profiles = auth_section.setdefault("profiles", {})
    profiles[profile_id] = {
        "provider": PROVIDER_ID,
        "mode": "oauth",
        "email": email,
    }
    _write_json(config_path, config)


# =============================================================================
# .env persistence
# =============================================================================

_CODEX_TOKEN_ENV_KEYS = ["OPENAI_CODEX_ACCESS_TOKEN"]


def _project_env_path() -> str:
    return os.getenv("DOTENV_PATH") or str(Path(__file__).resolve().parent.parent / ".env")


def _openclaw_env_path() -> str:
    return str(Path.home() / ".openclaw" / ".env")


def _upsert_env_key(env_path: str, key: str, value: str) -> None:
    if not os.path.isfile(env_path):
        os.makedirs(os.path.dirname(env_path), exist_ok=True)
        with open(env_path, "w") as f:
            f.write(f'{key}="{value}"\n')
        return
    lines: list[str] = []
    found = False
    with open(env_path, "r") as f:
        for raw in f:
            if raw.strip().startswith(f"{key}="):
                lines.append(f'{key}="{value}"\n')
                found = True
            else:
                lines.append(raw)
    if not found:
        lines.append(f'\n{key}="{value}"\n')
    with open(env_path, "w") as f:
        f.writelines(lines)


def _persist_token_to_env_files(access_token: str) -> None:
    for env_path in [_project_env_path(), _openclaw_env_path()]:
        for key in _CODEX_TOKEN_ENV_KEYS:
            try:
                _upsert_env_key(env_path, key, access_token)
            except OSError as e:
                print(f"[codex-setup] Failed to write {key} to {env_path}: {e}")


# =============================================================================
# Token exchange
# =============================================================================

async def _exchange_code_for_tokens(
    code: str,
    code_verifier: str,
    redirect_uri: str = DEVICE_CALLBACK_REDIRECT_URI,
) -> Dict[str, Any]:
    payload = {
        "grant_type": "authorization_code",
        "client_id": OPENAI_CLIENT_ID,
        "code": code,
        "code_verifier": code_verifier,
        "redirect_uri": redirect_uri,
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            OPENAI_TOKEN_URL,
            data=payload,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"OpenAI token exchange failed ({resp.status_code}): {resp.text[:500]}",
        )
    data = resp.json()
    for field in ("access_token", "refresh_token", "expires_in"):
        if field not in data:
            raise HTTPException(
                status_code=502,
                detail=f"OpenAI token response missing '{field}'",
            )
    return data


async def refresh_access_token(refresh_token: str) -> Dict[str, Any]:
    payload = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": OPENAI_CLIENT_ID,
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            OPENAI_TOKEN_URL,
            data=payload,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"OpenAI token refresh failed ({resp.status_code}): {resp.text[:500]}",
        )
    return resp.json()


# =============================================================================
# Router & endpoints
# =============================================================================

router = APIRouter(prefix="/codex", tags=["codex-setup"])


@router.post("/setup")
async def codex_setup():
    """
    Start the Codex device-code OAuth flow.

    Returns an SSE stream that emits:
      - {type: "user_code", code: "XXXX-XXXX", url: "https://auth.openai.com/codex/device"}
      - {type: "error", message: "..."}   on failure
      - {type: "done", email: "..."}      on success (credentials already persisted)

    The user must have "Device Code" enabled in ChatGPT security settings.
    """
    async def generate() -> AsyncGenerator[str, None]:
        print("[codex-setup] starting device code flow")

        try:
            # Step 1: request a user code from OpenAI
            async with httpx.AsyncClient(timeout=30.0) as client:
                uc_resp = await client.post(
                    DEVICE_USERCODE_URL,
                    json={"client_id": OPENAI_CLIENT_ID},
                )

            if uc_resp.status_code != 200:
                print(f"[codex-setup] usercode request failed ({uc_resp.status_code}): {uc_resp.text[:200]}")
                yield f"data: {json.dumps({'type': 'error', 'message': f'Failed to start device auth ({uc_resp.status_code}). Make sure Device Code login is enabled in your ChatGPT security settings.'})}\n\n"
                return

            uc_data = uc_resp.json()
            device_auth_id: str = uc_data["device_auth_id"]
            user_code: str = uc_data.get("usercode") or uc_data.get("user_code", "")
            poll_interval: int = max(int(uc_data.get("interval", MIN_POLL_INTERVAL)), MIN_POLL_INTERVAL)

            print(f"[codex-setup] got user_code={user_code}, poll_interval={poll_interval}s")

            # Step 2: emit the code for the frontend to display
            yield f"data: {json.dumps({'type': 'user_code', 'code': user_code, 'url': DEVICE_VERIFICATION_URL})}\n\n"

            # Step 3: poll until user completes login or timeout
            deadline = time.monotonic() + CODEX_SETUP_TIMEOUT_SECONDS
            heartbeat_tick = 0

            while time.monotonic() < deadline:
                await asyncio.sleep(poll_interval)
                heartbeat_tick += 1

                async with httpx.AsyncClient(timeout=30.0) as client:
                    poll_resp = await client.post(
                        DEVICE_TOKEN_URL,
                        json={"device_auth_id": device_auth_id, "user_code": user_code},
                    )

                if poll_resp.status_code in (403, 404):
                    # Still waiting for user — keep polling
                    if heartbeat_tick % 4 == 0:
                        yield ": heartbeat\n\n"
                    continue

                if poll_resp.status_code != 200:
                    detail = poll_resp.text[:300]
                    print(f"[codex-setup] poll error ({poll_resp.status_code}): {detail}")
                    yield f"data: {json.dumps({'type': 'error', 'message': f'Device auth polling failed ({poll_resp.status_code})'})}\n\n"
                    return

                # User has completed login
                code_data = poll_resp.json()
                authorization_code: str = code_data["authorization_code"]
                code_verifier: str = code_data["code_verifier"]

                print(f"[codex-setup] authorization code received, exchanging for tokens")

                # Step 4: exchange authorization code for tokens
                try:
                    token_data = await _exchange_code_for_tokens(
                        code=authorization_code,
                        code_verifier=code_verifier,
                        redirect_uri=DEVICE_CALLBACK_REDIRECT_URI,
                    )
                except HTTPException as e:
                    yield f"data: {json.dumps({'type': 'error', 'message': e.detail})}\n\n"
                    return

                access_token: str = token_data["access_token"]
                refresh_token: str = token_data["refresh_token"]
                expires_in: int = token_data["expires_in"]
                expires_at = int(time.time() * 1000) + (expires_in * 1000)

                identity = resolve_codex_auth_identity(access_token)
                email = identity["email"]
                profile_name = identity["profile_name"]
                account_id = identity["account_id"]

                print(f"[codex-setup] token exchange success (email={email}, profile={profile_name})")

                try:
                    write_auth_credentials(
                        profile_name=profile_name,
                        access=access_token,
                        refresh=refresh_token,
                        expires=expires_at,
                        email=email,
                    )
                except OSError as e:
                    print(f"[codex-setup] WARNING: failed to write auth credentials: {e}")

                _persist_token_to_env_files(access_token)
                for key in _CODEX_TOKEN_ENV_KEYS:
                    os.environ[key] = access_token

                print("[codex-setup] credentials persisted, emitting done")
                yield f"data: {json.dumps({'type': 'done', 'email': email, 'profile_name': profile_name, 'account_id': account_id})}\n\n"
                return

            print("[codex-setup] device auth timed out")
            yield f"data: {json.dumps({'type': 'error', 'message': 'Device auth timed out (15 minutes). Please try again.'})}\n\n"

        except Exception as e:
            print(f"[codex-setup] unexpected error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )

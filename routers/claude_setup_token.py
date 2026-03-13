"""
Claude Code CLI setup-token flow: run `claude setup-token` and stream output;
support pasted OAuth code when redirect fails (e.g. user on different machine).

On success, persists CLAUDE_CODE_OAUTH_TOKEN to .env and sets it in process env.
"""

import asyncio
import json
import os
import re
import shutil
from pathlib import Path
from typing import Optional, AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel


# OAuth token in CLI output: full token is ~108 chars; CLI may wrap at 80 chars so we merge continuation lines
_OAUTH_TOKEN_PATTERN = re.compile(r"sk-ant-oat01-\S+")
_MIN_FULL_TOKEN_LEN = 100  # tokens are ~108 chars; if we get less, expect a continuation line
_CONTINUATION_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")  # next line is rest of token (no spaces)


def _strip_ansi(line: str) -> str:
    """Remove ANSI escape sequences from CLI output."""
    return re.sub(r"\x1b\[[0-9;]*[a-zA-Zm]?", "", line)


def _extract_oauth_token(line: str) -> Optional[str]:
    """If line contains a Claude OAuth token, return it; else None."""
    plain = _strip_ansi(line).strip()
    match = _OAUTH_TOKEN_PATTERN.search(plain)
    return match.group(0) if match else None


def _is_continuation_line(line: str) -> bool:
    """True if line looks like the rest of a wrapped token (no spaces, token chars only)."""
    plain = _strip_ansi(line).strip()
    return bool(plain and not plain.startswith("sk-ant-") and _CONTINUATION_PATTERN.fullmatch(plain))


def _merge_and_extract_token(partial: str, continuation: str) -> Optional[str]:
    """Merge partial token line + continuation line and return full token if valid."""
    combined = (partial + continuation).strip()
    match = _OAUTH_TOKEN_PATTERN.search(combined)
    return match.group(0) if match and len(match.group(0)) >= _MIN_FULL_TOKEN_LEN else None


def _default_env_path() -> str:
    """Project root .env (same repo as this router), so persist works regardless of cwd."""
    return os.getenv("DOTENV_PATH") or str(Path(__file__).resolve().parent.parent / ".env")


def _persist_token_to_env_file(token: str) -> None:
    """
    Add or update CLAUDE_CODE_OAUTH_TOKEN in .env so it persists across restarts.
    Also safe for token values that contain '=' or '#' by quoting.
    """
    env_path = _default_env_path()
    if not os.path.isfile(env_path):
        with open(env_path, "w") as f:
            f.write(f'CLAUDE_CODE_OAUTH_TOKEN="{token}"\n')
        return
    lines: list[str] = []
    key = "CLAUDE_CODE_OAUTH_TOKEN"
    found = False
    with open(env_path, "r") as f:
        for raw in f:
            if raw.strip().startswith(f"{key}="):
                lines.append(f'{key}="{token}"\n')
                found = True
            else:
                lines.append(raw)
    if not found:
        lines.append(f'\n{key}="{token}"\n')
    with open(env_path, "w") as f:
        f.writelines(lines)


def _resolve_claude_cli_path() -> str:
    """Resolve Claude CLI path from env or PATH (avoids circular import from server)."""
    path = (os.getenv("CLAUDE_CLI_PATH") or "claude").strip()
    if not os.path.isabs(path):
        found = shutil.which(path)
        if found:
            return found
    return path


CLAUDE_SETUP_TOKEN_TIMEOUT_SECONDS = int(os.getenv("CLAUDE_SETUP_TOKEN_TIMEOUT", "300"))

_setup_token_lock = asyncio.Lock()
_setup_token_process: Optional[asyncio.subprocess.Process] = None
_setup_token_stdin: Optional[asyncio.StreamWriter] = None


router = APIRouter(prefix="/claude", tags=["claude-setup-token"])


class ClaudeSetupTokenCallbackBody(BaseModel):
    """Body for pasting OAuth code when redirect fails (e.g. user on different machine)."""
    code: str  # Paste the full string from the browser, e.g. "code#state"


@router.post("/setup-token/callback")
async def claude_setup_token_callback(body: ClaudeSetupTokenCallbackBody):
    """
    Send the pasted OAuth code to the running `claude setup-token` process.

    When the user authorizes in the browser and the redirect fails (e.g. they're on a
    different machine), Anthropic shows a "Paste this into Claude Code" page. The user
    copies that string and the frontend sends it here; we write it to the CLI's stdin
    so the flow can complete and the token is streamed back.
    """
    global _setup_token_process, _setup_token_stdin
    async with _setup_token_lock:
        if _setup_token_stdin is None or _setup_token_process is None:
            raise HTTPException(
                status_code=409,
                detail="No setup-token session active. Start one with POST /claude/setup-token first.",
            )
        if _setup_token_process.returncode is not None:
            _setup_token_process = None
            _setup_token_stdin = None
            raise HTTPException(
                status_code=409,
                detail="Setup-token process already finished. Start a new session if needed.",
            )
        try:
            payload = (body.code.strip() + "\n").encode("utf-8")
            _setup_token_stdin.write(payload)
            await _setup_token_stdin.drain()
            return {"ok": True, "message": "Code sent to CLI"}
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            _setup_token_process = None
            _setup_token_stdin = None
            raise HTTPException(status_code=410, detail=f"Process stdin closed: {e}")


@router.post("/setup-token")
async def claude_setup_token():
    """
    Run `claude setup-token` and stream stdout/stderr to the client via SSE.

    The CLI prints an OAuth URL; the frontend shows it so the user can open it and
    authorize. If the redirect goes to the user's localhost (different machine),
    Anthropic shows "Paste this into Claude Code" — the user copies that string and
    the frontend sends it to POST /claude/setup-token/callback so we can complete the flow.
    """
    global _setup_token_process, _setup_token_stdin
    cli_path = _resolve_claude_cli_path()

    async def generate() -> AsyncGenerator[str, None]:
        global _setup_token_process, _setup_token_stdin
        queue: asyncio.Queue = asyncio.Queue()
        process = None
        stdin_writer = None
        token_buffer: Optional[str] = None

        async def read_stdout(proc: asyncio.subprocess.Process) -> None:
            if proc.stdout is None:
                return
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").rstrip("\n")
                await queue.put(("stdout", text))

        async def read_stderr(proc: asyncio.subprocess.Process) -> None:
            if proc.stderr is None:
                return
            while True:
                line = await proc.stderr.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").rstrip("\n")
                await queue.put(("stderr", text))

        async def wait_done(proc: asyncio.subprocess.Process) -> None:
            returncode = await proc.wait()
            await queue.put(("done", returncode))

        def clear_session() -> None:
            async def _clear():
                global _setup_token_process, _setup_token_stdin
                async with _setup_token_lock:
                    if _setup_token_process is process:
                        _setup_token_process = None
                        _setup_token_stdin = None
            asyncio.create_task(_clear())

        try:
            process = await asyncio.create_subprocess_exec(
                cli_path,
                "setup-token",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.PIPE,
                env=os.environ.copy(),
            )
            stdin_writer = process.stdin
            async with _setup_token_lock:
                _setup_token_process = process
                _setup_token_stdin = stdin_writer

            asyncio.create_task(read_stdout(process))
            asyncio.create_task(read_stderr(process))
            asyncio.create_task(wait_done(process))

            while True:
                try:
                    item = await asyncio.wait_for(
                        queue.get(), timeout=CLAUDE_SETUP_TOKEN_TIMEOUT_SECONDS
                    )
                except asyncio.TimeoutError:
                    if process and process.returncode is None:
                        process.kill()
                    clear_session()
                    yield f"data: {json.dumps({'type': 'error', 'error': 'Setup timed out'})}\n\n"
                    break
                kind, value = item
                if kind == "done":
                    clear_session()
                    yield f"data: {json.dumps({'type': 'done', 'returncode': value})}\n\n"
                    break
                # On success, CLI may print the OAuth token (~108 chars); CLI often wraps at 80 chars
                if kind == "stdout":
                    def persist_token(t: str) -> None:
                        os.environ["CLAUDE_CODE_OAUTH_TOKEN"] = t
                        try:
                            _persist_token_to_env_file(t)
                            print(f"[claude-setup-token] Token persisted to .env (len={len(t)})")
                        except OSError as e:
                            print(f"[claude-setup-token] Token in env; .env write failed: {e}")

                    if token_buffer is not None:
                        if _is_continuation_line(value):
                            merged = _merge_and_extract_token(token_buffer, value)
                            if merged:
                                persist_token(merged)
                                token_buffer = None
                        else:
                            if token_buffer.startswith("sk-ant-oat01-") and len(token_buffer) >= _MIN_FULL_TOKEN_LEN:
                                persist_token(token_buffer)
                            token_buffer = None

                    if token_buffer is None:
                        token = _extract_oauth_token(value)
                        if token:
                            if len(token) >= _MIN_FULL_TOKEN_LEN:
                                persist_token(token)
                            else:
                                token_buffer = token
                yield f"data: {json.dumps({'type': kind, 'line': value})}\n\n"
        except FileNotFoundError:
            clear_session()
            yield f"data: {json.dumps({'type': 'error', 'error': f'CLI not found: {cli_path}'})}\n\n"
        except Exception as e:
            clear_session()
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )

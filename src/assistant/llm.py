"""LLM access configured entirely through environment variables, with a demo fallback.

  LLM_PROVIDER   anthropic (default) | openai (any OpenAI-compatible endpoint) | mock
  LLM_API_KEY    secret key; if empty/placeholder the assistant runs in demo (mock) mode
  LLM_MODEL      model id (defaults per provider)
  LLM_BASE_URL   optional override, e.g. a proxy or OpenAI-compatible server
  LLM_MAX_TOKENS optional, default 900
  LLM_TIMEOUT    optional seconds, default 60
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Optional

DEFAULT_MODELS = {"anthropic": "claude-sonnet-5", "openai": "gpt-4o-mini"}
DEFAULT_URLS = {"anthropic": "https://api.anthropic.com", "openai": "https://api.openai.com/v1"}
_PLACEHOLDERS = {"", "your_api_key", "your-api-key", "changeme", "xxx", "sk-...", "none", "null"}


class LLMError(RuntimeError):
    pass


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "mock"
    api_key: str = ""
    model: str = ""
    base_url: str = ""
    max_tokens: int = 900
    timeout: int = 60

    @property
    def is_live(self) -> bool:
        return self.provider in DEFAULT_URLS and bool(self.api_key)

    def describe(self) -> str:
        return f"{self.provider} / {self.model}" if self.is_live else "Demo mode (no LLM key configured)"

    @classmethod
    def from_env(cls, env: Optional[dict] = None) -> "LLMConfig":
        env = os.environ if env is None else env
        provider = (env.get("LLM_PROVIDER") or "anthropic").strip().lower()
        key = (env.get("LLM_API_KEY") or "").strip()
        if provider not in DEFAULT_URLS or key.lower() in _PLACEHOLDERS:
            return cls(provider="mock")
        return cls(provider=provider, api_key=key,
                   model=(env.get("LLM_MODEL") or DEFAULT_MODELS[provider]).strip(),
                   base_url=(env.get("LLM_BASE_URL") or DEFAULT_URLS[provider]).rstrip("/"),
                   max_tokens=int(env.get("LLM_MAX_TOKENS") or 900), timeout=int(env.get("LLM_TIMEOUT") or 60))


def _http_post(url: str, headers: dict, payload: dict, timeout: int) -> dict:
    if not url.lower().startswith(("https://", "http://")):        # LLM_BASE_URL is operator-supplied: refuse file:, ftp:, etc.
        raise LLMError("LLM_BASE_URL must start with https:// (or http:// for a local gateway)")
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310 - scheme validated above
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raise LLMError(f"HTTP {e.code} from LLM provider") from None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        raise LLMError(f"LLM request failed: {type(e).__name__}") from None


def complete(cfg: LLMConfig, system: str, user: str, http_post: Callable = _http_post) -> str:
    """Send one request to the configured provider and return the text reply."""
    return complete_with_usage(cfg, system, user, http_post)[0]


def complete_with_usage(cfg: LLMConfig, system: str, user: str, http_post: Callable = _http_post) -> tuple[str, Optional[dict]]:
    """Like complete() but also returns provider-reported token usage ({input,output,total}_tokens) or None."""
    if not cfg.is_live:
        raise LLMError("LLM is not configured")
    if cfg.provider == "anthropic":
        data = http_post(f"{cfg.base_url}/v1/messages",
                         {"x-api-key": cfg.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                         {"model": cfg.model, "max_tokens": cfg.max_tokens, "system": system,
                          "messages": [{"role": "user", "content": user}]}, cfg.timeout)
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        u = data.get("usage") or {}
        usage = _usage(u.get("input_tokens"), u.get("output_tokens"))
    else:
        data = http_post(f"{cfg.base_url}/chat/completions",
                         {"Authorization": f"Bearer {cfg.api_key}", "content-type": "application/json"},
                         {"model": cfg.model, "max_tokens": cfg.max_tokens, "temperature": 0,
                          "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
                         cfg.timeout)
        text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
        u = data.get("usage") or {}
        usage = _usage(u.get("prompt_tokens"), u.get("completion_tokens"), u.get("total_tokens"))
    if not text.strip():
        raise LLMError("LLM returned an empty response")
    return text.strip(), usage


def _usage(inp, out, total=None) -> Optional[dict]:
    if not isinstance(inp, int) and not isinstance(out, int):
        return None
    inp, out = int(inp or 0), int(out or 0)
    return {"input_tokens": inp, "output_tokens": out, "total_tokens": int(total) if isinstance(total, int) else inp + out}

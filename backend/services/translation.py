# hindi-dubbing-platform/backend/services/translation.py

"""Context-aware dialogue translation: OpenRouter (cloud) or Ollama (local), with optional fallback.

The LLM always answers in JSON keyed by segment id, so a missing, merged or reordered line is
detected instead of silently shifting every following line onto the wrong timestamp.
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from dataclasses import dataclass
from typing import Optional

import httpx

from backend.app.config import Settings, TranslationEngine

log = logging.getLogger("dubbing.translation")

PROMPT_VERSION = 3
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class TranslationError(RuntimeError):
    pass


@dataclass(frozen=True)
class SourceLine:
    id: str
    speaker: str
    text: str
    duration: float


@dataclass(frozen=True)
class ContextLine:
    speaker: str
    source: str
    hindi: str


def system_prompt(source_language: str) -> str:
    return (
        f"Translate the following character dialogue from {source_language} into natural, idiomatic Hindi "
        "suitable for film dubbing. Preserve the intensity and length. Do not summarize.\n\n"
        "Rules:\n"
        "- Write in Devanagari script only. Use the spoken Hindi of Hindi film/TV dubbing, not formal or "
        "Sanskritised Hindi; common English loanwords are fine when written in Devanagari.\n"
        "- Each line has the speaking time of the original in seconds. The Hindi must be speakable in the same "
        "time: match the original's length, never pad, never drop meaning.\n"
        "- Keep each speaker's register and emotion (anger, fear, sarcasm, tenderness) and keep forms of address "
        "(आप / तुम / तू) consistent per speaker pair. Keep interjections, hesitations and repetitions.\n"
        "- Keep character and place names; transliterate them into Devanagari.\n"
        "- Write numbers, dates, times and units as spoken Hindi words, never digits.\n"
        "- 'context' lines are earlier dialogue for continuity only. Do not translate or return them.\n"
        "- Respond with JSON only, exactly: {\"translations\": [{\"id\": \"<id>\", \"text\": \"<Hindi>\"}]} with "
        "one entry per input line id, in the same order."
    )


def shorten_system_prompt(source_language: str) -> str:
    return (
        "You edit Hindi film-dubbing lines so they fit the on-screen speaking time. Rewrite the Hindi line more "
        f"concisely while keeping the full meaning of the {source_language} original, its emotion and its "
        "register. Devanagari only, spoken film Hindi, numbers as words. Do not summarize away information that "
        "matters to the scene. Respond with JSON only: {\"text\": \"<shorter Hindi line>\"}"
    )


class _LLMClient:
    name: str

    def complete_json(self, system: str, user: str) -> dict:
        raise NotImplementedError

    def close(self) -> None:
        pass


class OpenRouterClient(_LLMClient):
    name = "openrouter"

    def __init__(self, settings: Settings):
        key = settings.openrouter_api_key.get_secret_value() if settings.openrouter_api_key else ""
        if not key:
            raise TranslationError("OPENROUTER_API_KEY is not configured.")
        self.model = settings.openrouter_model
        self.temperature = settings.translation_temperature
        self.retries = settings.llm_max_retries
        self._http = httpx.Client(
            base_url=settings.openrouter_base_url.rstrip("/"),
            timeout=httpx.Timeout(settings.llm_timeout_seconds, connect=20.0),
            headers={
                "Authorization": f"Bearer {key}",
                "X-Title": "Hindi AI Dubbing Platform",
            },
        )

    def complete_json(self, system: str, user: str) -> dict:
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        body = _post_with_retry(self._http, "/chat/completions", payload, self.retries, self.name)
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise TranslationError(f"OpenRouter returned an unexpected body: {str(body)[:300]}") from exc
        return _parse_json_object(content)

    def close(self) -> None:
        self._http.close()


class OllamaClient(_LLMClient):
    name = "ollama"

    def __init__(self, settings: Settings):
        self.model = settings.ollama_model
        self.temperature = settings.translation_temperature
        self.num_ctx = settings.ollama_num_ctx
        self.retries = settings.llm_max_retries
        self._http = httpx.Client(
            base_url=settings.ollama_url.rstrip("/"),
            timeout=httpx.Timeout(settings.llm_timeout_seconds, connect=10.0),
        )

    def complete_json(self, system: str, user: str) -> dict:
        payload = {
            "model": self.model,
            "system": system,
            "prompt": user,
            "format": "json",
            "stream": False,
            "options": {"temperature": self.temperature, "num_ctx": self.num_ctx},
        }
        body = _post_with_retry(self._http, "/api/generate", payload, self.retries, self.name)
        if "response" not in body:
            raise TranslationError(f"Ollama returned an unexpected body: {str(body)[:300]}")
        return _parse_json_object(body["response"])

    def close(self) -> None:
        self._http.close()


def build_client(engine: TranslationEngine, settings: Settings) -> _LLMClient:
    if engine == TranslationEngine.OPENROUTER:
        return OpenRouterClient(settings)
    return OllamaClient(settings)


def engine_fingerprint(engine: TranslationEngine, settings: Settings) -> dict:
    model = settings.openrouter_model if engine == TranslationEngine.OPENROUTER else settings.ollama_model
    return {"engine": engine.value, "model": model, "temperature": settings.translation_temperature,
            "prompt_version": PROMPT_VERSION}


class Translator:
    """Batches lines with rolling context; falls back to a second engine and finally to halving batches."""

    def __init__(self, settings: Settings, engine: TranslationEngine):
        self.settings = settings
        self.clients: list[_LLMClient] = [build_client(engine, settings)]
        fallback = settings.translation_fallback_engine
        if fallback and fallback != engine:
            try:
                self.clients.append(build_client(fallback, settings))
            except TranslationError as exc:
                log.warning("Fallback translation engine unavailable: %s", exc)

    def close(self) -> None:
        for c in self.clients:
            c.close()

    def __enter__(self) -> "Translator":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ public API
    def translate(self, lines: list[SourceLine], source_language: str, progress=None) -> dict[str, str]:
        result: dict[str, str] = {}
        context: list[ContextLine] = []
        size = self.settings.translation_batch_size
        n_ctx = self.settings.translation_context_lines
        for i in range(0, len(lines), size):
            batch = lines[i:i + size]
            translated = self._translate_batch(batch, context[-n_ctx:] if n_ctx else [], source_language)
            result.update(translated)
            context.extend(ContextLine(l.speaker, l.text, translated[l.id]) for l in batch)
            if progress:
                progress(min(1.0, (i + len(batch)) / len(lines)))
        return result

    def shorten(self, source_text: str, hindi: str, target_chars: int, duration: float, source_language: str) -> str:
        user = json.dumps({
            "original": source_text,
            "current_hindi": hindi,
            "current_length_chars": len(hindi),
            "target_max_chars": target_chars,
            "speaking_time_seconds": round(duration, 2),
        }, ensure_ascii=False)
        data = self._ask(shorten_system_prompt(source_language), user)
        text = str(data.get("text", "")).strip()
        if not text or not _DEVANAGARI.search(text):
            raise TranslationError("Shorten request returned no Hindi text.")
        return text

    # ------------------------------------------------------------------ internals
    def _translate_batch(self, batch: list[SourceLine], context: list[ContextLine], source_language: str) -> dict[str, str]:
        user = json.dumps({
            "context": [{"speaker": c.speaker, "original": c.source, "hindi": c.hindi} for c in context],
            "lines": [{"id": l.id, "speaker": l.speaker, "seconds": round(l.duration, 2), "text": l.text} for l in batch],
        }, ensure_ascii=False)
        try:
            data = self._ask(system_prompt(source_language), user)
            return _validate_batch(data, batch)
        except TranslationError as exc:
            if len(batch) == 1:
                raise TranslationError(f"Could not translate segment {batch[0].id}: {exc}") from exc
            mid = len(batch) // 2
            log.warning("Batch of %d failed validation (%s); splitting.", len(batch), exc)
            first = self._translate_batch(batch[:mid], context, source_language)
            n_ctx = self.settings.translation_context_lines
            ctx2 = context + [ContextLine(l.speaker, l.text, first[l.id]) for l in batch[:mid]]
            second = self._translate_batch(batch[mid:], ctx2[-n_ctx:] if n_ctx else [], source_language)
            return {**first, **second}

    def _ask(self, system: str, user: str) -> dict:
        errors = []
        for client in self.clients:
            try:
                return client.complete_json(system, user)
            except TranslationError as exc:
                errors.append(f"{client.name}: {exc}")
                log.warning("Translation engine %s failed: %s", client.name, exc)
        raise TranslationError("; ".join(errors))


def _validate_batch(data: dict, batch: list[SourceLine]) -> dict[str, str]:
    items = data.get("translations")
    if not isinstance(items, list):
        raise TranslationError("response has no 'translations' list")
    by_id = {}
    for item in items:
        if isinstance(item, dict) and "id" in item:
            by_id[str(item["id"])] = str(item.get("text", "")).strip()
    out = {}
    for line in batch:
        text = by_id.get(line.id, "")
        if not text:
            raise TranslationError(f"missing translation for {line.id}")
        if not _DEVANAGARI.search(text):
            raise TranslationError(f"translation for {line.id} is not in Devanagari: {text[:60]!r}")
        out[line.id] = text
    return out


def _parse_json_object(content: str) -> dict:
    text = (content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as first:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise TranslationError(f"LLM did not return JSON: {text[:200]!r}") from first
        try:
            obj = json.loads(text[start:end + 1])
        except json.JSONDecodeError as exc:
            raise TranslationError(f"LLM returned malformed JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise TranslationError("LLM JSON is not an object")
    return obj


def _post_with_retry(http: httpx.Client, url: str, payload: dict, retries: int, name: str) -> dict:
    last: Optional[str] = None
    for attempt in range(1, retries + 1):
        try:
            resp = http.post(url, json=payload)
        except httpx.TransportError as exc:
            last = f"transport error: {exc}"
        else:
            if resp.status_code == 200:
                try:
                    body = resp.json()
                except ValueError:
                    last = "non-JSON HTTP body"
                else:
                    if isinstance(body, dict) and body.get("error"):
                        last = f"API error: {str(body['error'])[:300]}"
                    else:
                        return body
            elif resp.status_code in _RETRYABLE_STATUS:
                last = f"HTTP {resp.status_code}: {resp.text[:200]}"
                retry_after = resp.headers.get("Retry-After")
                if retry_after and retry_after.isdigit() and attempt < retries:
                    time.sleep(min(float(retry_after), 60.0))
                    continue
            else:
                raise TranslationError(f"{name} HTTP {resp.status_code}: {resp.text[:300]}")
        if attempt < retries:
            delay = min(30.0, 2 ** attempt) + random.uniform(0, 1)
            log.warning("%s attempt %d/%d failed (%s); retrying in %.1fs", name, attempt, retries, last, delay)
            time.sleep(delay)
    raise TranslationError(f"{name} failed after {retries} attempts: {last}")

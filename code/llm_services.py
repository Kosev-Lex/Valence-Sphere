"""Small, purpose-specific LLM services used by the GUI.

The learning Answerer remains deliberately concise.  These services use their
own prompts so constituent analysis can return JSON and Reasoning can behave
like a normal conversational assistant without contaminating concept facts.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from constants import DEFAULT_OPENAI_MODEL


class LLMServiceError(RuntimeError):
    pass


@dataclass
class LLMResponse:
    text: str
    model: str
    usage: dict[str, Any]


class LLMClient:
    def __init__(self, model: str | None = None, temperature: float = 0.2):
        self.model = model or os.getenv("VS_MODEL") or DEFAULT_OPENAI_MODEL
        self.temperature = temperature
        key = os.getenv("VS_API") or os.getenv("OPENAI_API_KEY")
        if not key:
            raise LLMServiceError("Set VS_API or OPENAI_API_KEY before using the API.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMServiceError("Install the openai package before using the API.") from exc
        kwargs: dict[str, Any] = {"api_key": key}
        base_url = os.getenv("VS_BASE_URL")
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)

    def complete(self, system: str, user: str, *, temperature: float | None = None) -> LLMResponse:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=self.temperature if temperature is None else temperature,
            )
        except Exception as exc:
            raise LLMServiceError(str(exc)) from exc
        text = (response.choices[0].message.content or "").strip()
        usage = getattr(response, "usage", None)
        return LLMResponse(
            text=text,
            model=self.model,
            usage={
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0),
            },
        )


class ConstituentAnalysisService:
    SYSTEM = """You are a precise grammatical and semantic analyst.

Analyze the ACCEPTED ANSWER, not the question.

Identify only the words and phrases carrying the answer's important meaning. Ignore articles, routine auxiliaries,
filler, repetition, and structurally necessary words that add no important semantic content.

Use the QUESTION only to understand what information in the answer is relevant.

Return JSON only, with exactly these keys:
core_meaning, nouns, verbs, adjectives, adverbs, noun_phrases, verb_phrases, qualifiers, relations.

core_meaning must be one concise string summarising the operative meaning. Every other field must be an array of
strings. Classify each constituent according to its function in the accepted answer. Keep meaningful multi-word
expressions intact. Do not invent information."""

    def __init__(self, client: LLMClient | None = None):
        self.client = client or LLMClient(temperature=0.0)

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I | re.S)
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            match = re.search(r"\{.*\}", cleaned, flags=re.S)
            if not match:
                raise LLMServiceError("The API did not return valid constituent JSON.") from exc
            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError as nested:
                raise LLMServiceError("The API did not return valid constituent JSON.") from nested
        if not isinstance(payload, dict):
            raise LLMServiceError("Constituent analysis must be a JSON object.")
        string_keys = {"core_meaning"}
        list_keys = {
            "nouns", "verbs", "adjectives", "adverbs", "noun_phrases",
            "verb_phrases", "qualifiers", "relations",
        }
        missing = (string_keys | list_keys) - payload.keys()
        if missing:
            raise LLMServiceError("Constituent JSON is missing: " + ", ".join(sorted(missing)))
        if not isinstance(payload["core_meaning"], str):
            raise LLMServiceError("core_meaning must be a string.")
        if any(not isinstance(payload[key], list) or
               any(not isinstance(value, str) for value in payload[key]) for key in list_keys):
            raise LLMServiceError("Constituent category fields must be arrays of strings.")
        return {key: payload[key] for key in (
            "core_meaning", "nouns", "verbs", "adjectives", "adverbs", "noun_phrases",
            "verb_phrases", "qualifiers", "relations",
        )}

    def analyze(self, question: str, answer: str) -> tuple[dict[str, Any], LLMResponse]:
        response = self.client.complete(
            self.SYSTEM,
            f"QUESTION:\n{question.strip()}\n\nANSWER:\n{answer.strip()}",
            temperature=0.0,
        )
        return self._parse_json(response.text), response

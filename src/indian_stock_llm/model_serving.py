from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol
from urllib import request


@dataclass(frozen=True)
class ModelResponse:
    answer: str
    model_name: str
    latency_mode: str


class ModelBackend(Protocol):
    def generate(self, prompt: str, timeout_seconds: float) -> ModelResponse: ...


@dataclass(frozen=True)
class TemplateModelBackend:
    model_name: str = "template-composer"

    def generate(self, prompt: str, timeout_seconds: float) -> ModelResponse:
        _ = timeout_seconds
        return ModelResponse(answer=prompt, model_name=self.model_name, latency_mode="deterministic")


@dataclass(frozen=True)
class HttpModelBackend:
    endpoint: str
    api_key: str | None = None
    provider: str = "generic"
    model: str | None = None
    model_name: str = "remote-llm"

    def generate(self, prompt: str, timeout_seconds: float) -> ModelResponse:
        provider = self.provider.strip().lower()
        if provider in {"openai", "azure_openai"}:
            payload: dict[str, object] = {
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
            }
            if self.model:
                payload["model"] = self.model
        else:
            payload = {"prompt": prompt}
        req = request.Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        if self.api_key:
            header = "Authorization" if provider in {"openai", "azure_openai"} else "X-API-Key"
            req.add_header(header, self.api_key)
        with request.urlopen(req, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
        answer: str | None = None
        if isinstance(payload, dict):
            if provider in {"openai", "azure_openai"}:
                choices = payload.get("choices", [])
                if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                    message = choices[0].get("message", {})
                    if isinstance(message, dict):
                        content = message.get("content")
                        if isinstance(content, str):
                            answer = content
            else:
                answer = payload.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("model backend returned empty answer")
        return ModelResponse(answer=answer.strip(), model_name=self.model_name, latency_mode="inference")


class ModelOrchestrator:
    def __init__(
        self,
        primary: ModelBackend,
        fallback: ModelBackend | None = None,
        timeout_seconds: float = 2.5,
    ):
        self.primary = primary
        self.fallback = fallback or TemplateModelBackend()
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def compose_prompt(
        *,
        query: str,
        intent: str,
        category: str,
        context_text: str,
        citations: tuple[str, ...],
        deterministic_note: str,
        policy_disclaimer: str,
        readiness_note: str,
    ) -> str:
        citation_text = ", ".join(citations) if citations else "none"
        return (
            f"Intent: {intent}\n"
            f"Category: {category}\n"
            f"User query: {query.strip()}\n"
            f"Grounding context:\n{context_text}\n"
            f"Citations: {citation_text}\n"
            f"Readiness: {readiness_note}\n"
            f"Deterministic checks: {deterministic_note or 'none'}\n"
            "Answer with risk-aware language, cite only provided sources, and avoid guaranteed-return claims.\n"
            f"Compliance disclaimer: {policy_disclaimer}\n"
        )

    @staticmethod
    def enforce_citation_controls(answer: str, citations: tuple[str, ...], require_citations: bool) -> str:
        if not require_citations:
            return answer
        if not citations:
            return "Insufficient grounding: no trusted citations available for this answer."
        return answer

    def generate(self, prompt: str, *, require_citations: bool, citations: tuple[str, ...]) -> ModelResponse:
        try:
            response = self.primary.generate(prompt, timeout_seconds=self.timeout_seconds)
        except Exception:
            response = self.fallback.generate(prompt, timeout_seconds=self.timeout_seconds)
        answer = self.enforce_citation_controls(response.answer, citations=citations, require_citations=require_citations)
        return ModelResponse(answer=answer, model_name=response.model_name, latency_mode=response.latency_mode)

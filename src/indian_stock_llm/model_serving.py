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
        return ModelResponse(
            answer=self._compose_answer(prompt),
            model_name=self.model_name,
            latency_mode="deterministic",
        )

    @staticmethod
    def _extract_field(prompt: str, label: str) -> str:
        for line in prompt.splitlines():
            if line.startswith(f"{label}:"):
                return line.split(":", 1)[1].strip()
        return ""

    @classmethod
    def _extract_context_lines(cls, prompt: str) -> list[str]:
        marker = "Grounding context:\n"
        if marker not in prompt:
            return []
        tail = prompt.split(marker, 1)[1]
        stop_markers = ("\nCitations:", "\nReadiness:", "\nDeterministic checks:", "\nCompliance disclaimer:")
        end = len(tail)
        for stop in stop_markers:
            idx = tail.find(stop)
            if idx != -1:
                end = min(end, idx)
        context_block = tail[:end].strip()
        lines = [line.strip() for line in context_block.splitlines() if line.strip().startswith("- ")]
        return lines

    @classmethod
    def _compose_answer(cls, prompt: str) -> str:
        intent = cls._extract_field(prompt, "Intent") or "general_query"
        category = cls._extract_field(prompt, "Category") or "stocks"
        query = cls._extract_field(prompt, "User query") or "User query not provided."
        readiness = cls._extract_field(prompt, "Readiness") or "not available"
        deterministic = cls._extract_field(prompt, "Deterministic checks") or "none"
        disclaimer = cls._extract_field(prompt, "Compliance disclaimer")
        context_lines = cls._extract_context_lines(prompt)
        if context_lines:
            context_summary = "\n".join(context_lines[:3])
        else:
            context_summary = "- No grounded context was available."
        guidance_by_intent = {
            "stock_analysis": "Focus on trend, momentum, volume confirmation, valuation context, and event risk.",
            "fundamentals": "Focus on earnings quality, valuation metrics, balance-sheet strength, and sector positioning.",
            "events_news": "Focus on regulatory impact, management commentary, and timeline of material disclosures.",
            "prediction": "Focus on scenarios and risk factors; avoid directional certainty.",
            "market_calculations": "Use deterministic outputs as checks and validate assumptions with live market data.",
        }
        guidance = guidance_by_intent.get(intent, "Prioritize grounded facts and validate with live NSE/BSE data.")
        lines = [
            f"Query focus: {query}",
            f"Intent-category mapping: {intent} / {category}",
            f"Data readiness snapshot: {readiness}",
            "Grounded highlights:",
            context_summary,
        ]
        if deterministic.lower() != "none":
            lines.append(f"Deterministic checks: {deterministic}")
        lines.append(f"Analysis guidance: {guidance}")
        if disclaimer:
            lines.append(f"Compliance note: {disclaimer}")
        return "\n".join(lines)


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

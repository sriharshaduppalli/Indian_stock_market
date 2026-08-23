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
    def _extract_market_context(cls, prompt: str) -> dict:
        marker = "Live market context JSON:\n"
        if marker not in prompt:
            return {}
        tail = prompt.split(marker, 1)[1]
        stop = tail.find("\nGrounding context:")
        block = (tail if stop < 0 else tail[:stop]).strip()
        if not block or block == "none":
            return {}
        try:
            import json

            data = json.loads(block)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    @classmethod
    def _compose_answer(cls, prompt: str) -> str:
        intent = cls._extract_field(prompt, "Intent") or "general_query"
        category = cls._extract_field(prompt, "Category") or "stocks"
        query = cls._extract_field(prompt, "User query")
        context_lines = cls._extract_context_lines(prompt)
        market_context = cls._extract_market_context(prompt)
        deterministic = ""
        marker = "Deterministic checks:\n"
        if marker in prompt:
            block = prompt.split(marker, 1)[1]
            stop = block.find("\nCompliance disclaimer:")
            deterministic = (block if stop < 0 else block[:stop]).strip()
            if deterministic == "none":
                deterministic = ""

        try:
            from .answer_composer import compose_structured_answer

            structured = compose_structured_answer(
                query=query,
                intent=intent,
                market_context=market_context,
                context_lines=context_lines,
                deterministic=deterministic,
            )
            if structured:
                return structured

            # Only soft-retry bare general_query + symbol. Never overwrite a
            # focused intent (news / fundamentals / prediction) with full analysis.
            symbol = ""
            if isinstance(market_context, dict):
                symbol = str(market_context.get("symbol") or "").strip().upper()
            if symbol and intent in {"general_query", ""}:
                try:
                    from .answer_composer import resolve_stock_response_profile

                    profile = resolve_stock_response_profile(query, intent or "general_query")
                    retry_intent = {
                        "news": "events_news",
                        "sentiment": "events_news",
                        "quote": "price_action",
                        "technical": "stock_analysis",
                        "risks": "events_news",
                        "trade_plan": "price_action",
                        "prediction": "prediction",
                        "fundamentals": "fundamentals",
                        "calculations": "market_calculations",
                    }.get(profile, "stock_analysis")
                except Exception:
                    retry_intent = "stock_analysis"
                if retry_intent != intent:
                    structured = compose_structured_answer(
                        query=query,
                        intent=retry_intent,
                        market_context=market_context,
                        context_lines=context_lines,
                        deterministic=deterministic,
                    )
                    if structured:
                        return structured
        except Exception:
            pass

        # Never dump Live quote / Full math / Live enrich bullets as the user answer.
        symbol = ""
        if isinstance(market_context, dict):
            symbol = str(market_context.get("symbol") or "").strip().upper()
        if symbol:
            return (
                f"**{symbol}**\n\n"
                "I couldn't build a clean paper-practice summary from live feeds just now.\n\n"
                f"Try: “Should I buy {symbol}?”, “{symbol} sentiment”, or "
                f"“full math for {symbol}”.\n\n"
                "_Educational only — not investment advice._"
            )

        headers = {
            "market_calculations": "Indian market equations & calculations",
            "fundamentals": "Fundamental snapshot",
            "stock_analysis": "Stock analysis notes",
            "prediction": "Forecast framing (not a guarantee)",
            "portfolio": "Portfolio guidance",
            "derivatives": "F&O / derivatives notes",
            "events_news": "Market / regulatory context",
            "price_action": "Price-action notes",
            "compare": "Comparison notes",
            "sector_screen": "Sector screen",
            "overbought_check": "Overbought / oversold check",
            "market_literacy": "How the stock market works (beginner)",
            "general_query": "Indian market knowledge",
        }
        title = headers.get(intent, f"Indian market insight ({category})")
        parts = [f"**{title}**", ""]

        # Literacy / education only: keep short, non-raw snippets (skip live dumps).
        cleaned: list[str] = []
        for line in context_lines[:6]:
            text = line[2:].strip() if line.startswith("- ") else line
            low = text.lower()
            if any(
                bad in low
                for bad in (
                    "live quote",
                    "live enrich",
                    "full math for",
                    "quantitative + trade plan",
                    "wilder rsi",
                    "data_degraded",
                )
            ):
                continue
            if "NSE market status" in text:
                text = text.split("NSE market status")[0].strip().rstrip(".")
            if text and len(text) < 280:
                cleaned.append(text)
        if cleaned:
            for text in cleaned[:4]:
                parts.append(f"• {text}")
        else:
            parts.append(
                "Ask about a specific NSE/BSE symbol (e.g. “KAYNES sentiment”) "
                "or a market concept (e.g. “what is RSI?”)."
            )

        parts.extend(
            [
                "",
                "_Grounded by BYSEL._",
            ]
        )
        return "\n".join(parts)

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
        market_context_json: str = "none",
    ) -> str:
        citation_text = ", ".join(citations) if citations else "none"
        return (
            f"Intent: {intent}\n"
            f"Category: {category}\n"
            f"User query: {query.strip()}\n"
            f"Live market context JSON:\n{market_context_json or 'none'}\n"
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

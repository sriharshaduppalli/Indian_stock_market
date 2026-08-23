from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread


def _learning_fields(query: str) -> dict:
    try:
        from .query_language import learning_query_fields

        return learning_query_fields(query)
    except Exception:
        text = (query or "").strip()
        return {
            "original": text,
            "normalized": text,
            "key": text.lower(),
            "language": "en",
            "is_telugu": False,
        }


class ContinualLearningManager:
    """Simple hook for capturing daily feedback used in later model updates."""

    def __init__(self, feedback_log_path: Path | None, async_logging: bool = True):
        self.feedback_log_path = feedback_log_path
        self._async_logging = async_logging
        self._queue: Queue[str] | None = None
        self._stop_event: Event | None = None
        self._worker: Thread | None = None
        if self.feedback_log_path is not None:
            self.feedback_log_path.parent.mkdir(parents=True, exist_ok=True)
            self.feedback_log_path.touch(exist_ok=True)
        if self.feedback_log_path is not None and async_logging:
            self._queue = Queue(maxsize=1_000)
            self._stop_event = Event()
            self._worker = Thread(target=self._run_worker, daemon=True)
            self._worker.start()

    def _append_line(self, line: str) -> None:
        if self.feedback_log_path is None:
            return
        self.feedback_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.feedback_log_path.open("a", encoding="utf-8") as fp:
            fp.write(line)

    def _run_worker(self) -> None:
        if self._queue is None or self._stop_event is None:
            return
        while not self._stop_event.is_set():
            try:
                line = self._queue.get(timeout=0.2)
            except Empty:
                continue
            try:
                self._append_line(line)
            finally:
                self._queue.task_done()

    def _write_line(self, line: str) -> None:
        if self._queue is not None:
            try:
                self._queue.put_nowait(line)
                return
            except Exception:
                pass
        self._append_line(line)

    def record_feedback(self, query: str, intent: str) -> None:
        if self.feedback_log_path is None:
            return

        ts = datetime.now(timezone.utc).isoformat()
        fields = _learning_fields(query)
        # Store the English learning key so Telugu + English repeats merge.
        self._write_line(f"{ts}\t{intent}\t{fields['key']}\n")

    def daily_learning_summary(self) -> str:
        if self.feedback_log_path is None:
            return "Daily learning loop disabled: no feedback log path configured."

        query_count = 0
        if self.feedback_log_path.exists():
            with self.feedback_log_path.open("r", encoding="utf-8") as fp:
                query_count = sum(1 for _ in fp)
        return (
            f"Daily learning loop enabled: {query_count} feedback samples logged; data can be used to refresh retrieval, "
            "recalibrate prediction factors, and improve next-day responses."
        )

    @staticmethod
    def anonymize_query(query: str) -> str:
        return hashlib.sha256(query.strip().encode("utf-8")).hexdigest()

    def record_anonymized_feedback(self, query: str, intent: str) -> None:
        if self.feedback_log_path is None:
            return
        ts = datetime.now(timezone.utc).isoformat()
        payload = {"ts": ts, "intent": intent, "query_hash": self.anonymize_query(query)}
        self._write_line(json.dumps(payload, ensure_ascii=False) + "\n")

    def record_interaction(
        self,
        query: str,
        intent: str,
        confidence: float,
        citation_count: int,
        grounded: bool,
    ) -> None:
        """Append a richer JSONL interaction for closed-loop RAG learning.

        This improves retrieval/knowledge over time via feedback promotion into a
        learned KB — it is NOT full neural LLM fine-tuning / LoRA auto-train.
        Existing TSV ``record_feedback`` / anonymized paths are unchanged.
        """
        if self.feedback_log_path is None:
            return
        ts = datetime.now(timezone.utc).isoformat()
        fields = _learning_fields(query)
        payload = {
            "ts": ts,
            "kind": "interaction_v1",
            "query": str(fields["original"]),
            "query_key": str(fields["key"]),
            "language": str(fields["language"]),
            "intent": intent,
            "confidence": float(confidence),
            "citation_count": int(citation_count),
            "grounded": bool(grounded),
        }
        self._write_line(json.dumps(payload, ensure_ascii=False) + "\n")

    def feedback_metrics(self) -> dict[str, float]:
        if self.feedback_log_path is None or not self.feedback_log_path.exists():
            return {
                "feedback_samples": 0.0,
                "anonymized_samples": 0.0,
                "anonymized_ratio": 0.0,
            }
        total = 0
        anonymized = 0
        with self.feedback_log_path.open("r", encoding="utf-8") as fp:
            for line in fp:
                text = line.strip()
                if not text:
                    continue
                total += 1
                if text.startswith("{") and "query_hash" in text:
                    anonymized += 1
        ratio = (anonymized / total) if total else 0.0
        return {
            "feedback_samples": float(total),
            "anonymized_samples": float(anonymized),
            "anonymized_ratio": ratio,
        }

    def close(self, timeout_seconds: float = 1.0) -> None:
        if self._stop_event is None or self._worker is None:
            return
        self._stop_event.set()
        self._worker.join(timeout=timeout_seconds)


# Intent → representative tags used for knowledge-refresh prioritisation
_INTENT_TOP_TAGS: dict[str, list[str]] = {
    "fundamentals": ["fundamentals", "valuation"],
    "events_news": ["sebi", "regulation"],
    "market_calculations": ["calculation", "cagr"],
    "prediction": ["prediction", "forecast"],
    "stock_analysis": ["analysis", "technical"],
    "portfolio": ["portfolio", "risk"],
}


class DailyFeedbackAnalyzer:
    """Analyses daily feedback logs to surface intent distribution trends.

    Supports both raw TSV lines (``ts\\tintent\\tquery``) and JSON lines
    (``{"ts": ..., "intent": ..., "query_hash": ...}``).
    """

    def __init__(self, feedback_log_path: Path | None) -> None:
        self.feedback_log_path = feedback_log_path

    def analyze(self) -> dict[str, object]:
        """Return a summary dict with intent counts and top intents."""
        if self.feedback_log_path is None or not self.feedback_log_path.exists():
            return {
                "intent_counts": {},
                "total_samples": 0,
                "top_intents": [],
                "ready": False,
            }

        intent_counts: dict[str, int] = {}
        total = 0

        with self.feedback_log_path.open("r", encoding="utf-8") as fp:
            for line in fp:
                text = line.strip()
                if not text:
                    continue
                if text.startswith("{"):
                    try:
                        payload = json.loads(text)
                        intent = str(payload.get("intent", "unknown"))
                    except json.JSONDecodeError:
                        continue
                else:
                    parts = text.split("\t")
                    if len(parts) >= 2:
                        intent = parts[1]
                    else:
                        continue
                intent_counts[intent] = intent_counts.get(intent, 0) + 1
                total += 1

        top_intents = [i for i, _ in sorted(intent_counts.items(), key=lambda x: x[1], reverse=True)]
        return {
            "intent_counts": intent_counts,
            "total_samples": total,
            "top_intents": top_intents[:3],
            "ready": total >= 10,
        }

    def suggested_knowledge_refresh_tags(self) -> list[str]:
        """Return tags to prioritise when refreshing the knowledge-base index.

        Tags are derived from the top-3 most frequent intents in the feedback log.
        """
        analysis = self.analyze()
        tags: list[str] = []
        for intent in analysis.get("top_intents", []):
            tags.extend(_INTENT_TOP_TAGS.get(str(intent), []))
        # Deduplicate while preserving order
        seen: set[str] = set()
        result: list[str] = []
        for tag in tags:
            if tag not in seen:
                seen.add(tag)
                result.append(tag)
        return result[:6]


# Educational framing only — process coaching, never tips / guaranteed returns.
_INTENT_COACHING: dict[str, str] = {
    "fundamentals": (
        "Educational framing: when studying fundamentals, compare valuation ratios "
        "(P/E, P/B, ROE) across peers and cycles; treat single-period numbers as "
        "incomplete. This is process coaching, not a buy/sell tip and not a return guarantee."
    ),
    "events_news": (
        "Educational framing: for corporate/regulatory events, separate facts from "
        "commentary, note effective dates, and cross-check NSE/BSE/SEBI primary sources. "
        "This is process coaching — not investment advice or guaranteed outcomes."
    ),
    "market_calculations": (
        "Educational framing: verify inputs, units, and time windows before trusting any "
        "formula (CAGR, returns, indicators). Recalculate independently. Educational only — "
        "not a trading signal or guaranteed return."
    ),
    "prediction": (
        "Educational framing: forecasts are uncertain; list assumptions, scenarios, and "
        "invalidation criteria. Prefer risk framing over point targets. Not advice and "
        "not a promise of returns."
    ),
    "stock_analysis": (
        "Educational framing: use a checklist (business, valuation, technical context, "
        "risks) and write a journal entry before acting. Paper-practice first. Educational "
        "only — no guaranteed profits."
    ),
    "portfolio": (
        "Educational framing: size positions to risk tolerance, diversify thoughtfully, "
        "and review correlations. Process coaching for learning — not personalized advice "
        "or guaranteed returns."
    ),
    "price_action": (
        "Educational framing: describe structure (levels, volume context) without "
        "prescribing trades. Always note invalidation. Educational / paper-practice only."
    ),
    "paper_practice": (
        "Educational framing: journal setups, risk rules, and post-trade reviews in "
        "simulation before live capital. Process over tip certainty. Not SEBI RA advice."
    ),
    "general_query": (
        "Educational framing: clarify the topic (symbol, indicator, regulation, or "
        "concept), then ground answers in cited market knowledge. Informational only — "
        "not guaranteed returns."
    ),
}


def _slug_id(prefix: str, text: str) -> str:
    digest = hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _load_learned_items(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(data, list):
        return []
    return [row for row in data if isinstance(row, dict) and row.get("id")]


def _save_learned_items(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_id: dict[str, dict] = {}
    for row in items:
        item_id = str(row.get("id", "")).strip()
        if not item_id:
            continue
        by_id[item_id] = row
    path.write_text(json.dumps(list(by_id.values()), ensure_ascii=False, indent=2), encoding="utf-8")


class FeedbackLearningPipeline:
    """Promote frequent feedback + grounded answers into a learned knowledge JSON.

    Honest scope: this improves RAG retrieval/knowledge over time by writing
    educational KnowledgeItem-like records. It is NOT full neural LLM fine-tuning.
    """

    @staticmethod
    def promote_from_feedback_log(
        feedback_log_path: Path | None,
        learned_knowledge_path: Path | None,
        min_count: int = 3,
    ) -> int:
        """Promote frequent TSV queries into educational learned KB items.

        Returns the number of newly added items (0 if skipped / nothing new).
        """
        if feedback_log_path is None or learned_knowledge_path is None:
            return 0
        if not feedback_log_path.exists():
            return 0

        query_counts: dict[tuple[str, str], int] = {}
        telugu_keys: set[str] = set()
        with feedback_log_path.open("r", encoding="utf-8") as fp:
            for line in fp:
                text = line.strip()
                if not text:
                    continue
                intent = "general_query"
                raw_query = ""
                language = "en"
                if text.startswith("{"):
                    try:
                        payload = json.loads(text)
                    except json.JSONDecodeError:
                        continue
                    kind = str(payload.get("kind") or "")
                    if kind not in {"interaction_v1", "thumbs_v1", ""}:
                        # Skip anonymized hash-only rows.
                        if payload.get("query_hash") and not payload.get("query") and not payload.get("query_key"):
                            continue
                    intent = str(payload.get("intent") or "general_query").strip() or "general_query"
                    raw_query = str(payload.get("query_key") or payload.get("query") or "").strip()
                    language = str(payload.get("language") or "en")
                else:
                    parts = text.split("\t")
                    if len(parts) < 3:
                        continue
                    intent = parts[1].strip() or "general_query"
                    raw_query = parts[2].strip()
                if not raw_query:
                    continue
                fields = _learning_fields(raw_query)
                key = (str(fields["key"]), intent)
                query_counts[key] = query_counts.get(key, 0) + 1
                if fields["is_telugu"] or language in {"te", "te-en"}:
                    telugu_keys.add(str(fields["key"]))

        existing = _load_learned_items(learned_knowledge_path)
        existing_ids = {str(row["id"]) for row in existing}
        added = 0
        for (query_l, intent), count in query_counts.items():
            if count < max(1, int(min_count)):
                continue
            item_id = _slug_id("learned_fb", f"{intent}|{query_l}")
            if item_id in existing_ids:
                continue
            coaching = _INTENT_COACHING.get(intent, _INTENT_COACHING["general_query"])
            title = f"Frequent topic coaching: {query_l[:80]}"
            content = (
                f"Learners often ask about «{query_l}» (intent={intent}). {coaching}"
            )
            tags = list(_INTENT_TOP_TAGS.get(intent, [intent, "education"]))
            if "education" not in tags:
                tags.append("education")
            if query_l in telugu_keys:
                for extra in ("telugu", "tenglish", "indic"):
                    if extra not in tags:
                        tags.append(extra)
            existing.append(
                {
                    "id": item_id,
                    "title": title,
                    "content": content,
                    "tags": tags,
                    "source": "learned_feedback_v1",
                }
            )
            existing_ids.add(item_id)
            added += 1

        if added:
            _save_learned_items(learned_knowledge_path, existing)
        elif not learned_knowledge_path.exists():
            _save_learned_items(learned_knowledge_path, existing)
        return added

    @staticmethod
    def promote_grounded_answer(
        learned_knowledge_path: Path | None,
        query: str,
        intent: str,
        answer: str,
        citations: list[str] | tuple[str, ...] | None,
        confidence: float,
    ) -> bool:
        """Upsert a truncated grounded answer into the learned KB when quality gates pass.

        Gates: confidence ≥ 0.55 and at least one citation. Content is truncated (~1200 chars)
        and framed as educational retrieval material — not LoRA weights / tip guarantees.
        Raw grounding dumps and empty shells are rejected so the KB only learns readable answers.
        """
        if learned_knowledge_path is None:
            return False
        cite_list = [c for c in (citations or []) if c]
        if float(confidence) < 0.55 or not cite_list:
            return False
        cleaned_answer = (answer or "").strip()
        if not cleaned_answer:
            return False
        # Never promote the old raw grounding dump style into learned memory.
        dump_markers = (
            "Live quote snapshot for",
            "Full math for ",
            "| Wilder RSI(14)=",
            "TRADE PLAN action=",
            "• **Live quote ",
            "• **Quantitative + trade plan",
            "• **Live enrich ",
        )
        if any(m in cleaned_answer for m in dump_markers):
            return False
        if "\u0c24\u0c46\u0c32\u0c41\u0c17\u0c41 \u0c38\u0c3e\u0c30\u0c3e\u0c02\u0c36\u0c02" in cleaned_answer:
            # Learn the English composer output, not the Telugu overlay.
            return False
        if "**Indian market knowledge**" in cleaned_answer and "Live quote" in cleaned_answer:
            return False
        # Prefer richer structured answers for learning (plan + sentiment/news).
        max_chars = 1200
        if any(
            marker in cleaned_answer
            for marker in (
                "**Sentiment analysis:**",
                "**News & trends:**",
                "**Direct answer:**",
                "sentiment analysis",
            )
        ):
            max_chars = 1600
        truncated = cleaned_answer[:max_chars]
        if len(cleaned_answer) > max_chars:
            truncated = truncated.rstrip() + "…"
        fields = _learning_fields(query)
        learn_key = str(fields["key"]) or query.strip().lower()
        item_id = _slug_id("learned_ans", f"{intent}|{learn_key}")
        tags = list(_INTENT_TOP_TAGS.get(intent, [intent, "education"]))
        if "education" not in tags:
            tags.append("education")
        if fields["is_telugu"]:
            for extra in ("telugu", "tenglish", "indic"):
                if extra not in tags:
                    tags.append(extra)
        if "Sentiment analysis" in cleaned_answer or "News & trends" in cleaned_answer:
            for t in ("sentiment", "news", "live_context"):
                if t not in tags:
                    tags.append(t)
        item = {
            "id": item_id,
            "title": f"Grounded answer note: {learn_key[:72]}",
            "content": (
                f"Educational retrieved note for «{learn_key}» "
                f"(intent={intent}, citations={', '.join(cite_list[:5])}). "
                f"{truncated} "
                "Informational only — not investment advice and not a guaranteed return."
            ),
            "tags": tags,
            "source": "learned_answer_v1",
        }
        existing = _load_learned_items(learned_knowledge_path)
        by_id = {str(row["id"]): row for row in existing}
        by_id[item_id] = item
        _save_learned_items(learned_knowledge_path, list(by_id.values()))
        return True

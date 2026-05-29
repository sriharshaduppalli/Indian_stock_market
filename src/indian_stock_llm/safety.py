from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

PROMPT_INJECTION_PATTERNS = (
    "ignore previous",
    "system prompt",
    "jailbreak",
    "bypass safety",
    "developer message",
    "reveal hidden prompt",
)
PROHIBITED_ADVICE_PATTERNS = (
    "guaranteed return",
    "sure-shot",
    "all-in",
    "insider tip",
    "assured profit",
    "zero risk trade",
)


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    reason: str
    rule_id: str = "allow_default"


class SafetyPolicy:
    def __init__(self, audit_log_path: Path | None):
        self.audit_log_path = audit_log_path

    def evaluate(self, query: str) -> PolicyDecision:
        q = query.lower()
        if any(p in q for p in PROMPT_INJECTION_PATTERNS):
            decision = PolicyDecision(False, "Prompt-injection pattern detected", rule_id="prompt_injection_block")
            self._log(query, decision)
            return decision
        if any(p in q for p in PROHIBITED_ADVICE_PATTERNS):
            decision = PolicyDecision(False, "Prohibited unsafe-advice pattern detected", rule_id="unsafe_advice_block")
            self._log(query, decision)
            return decision
        decision = PolicyDecision(True, "Allowed with SEBI-aligned disclosure requirements", rule_id="allow_disclosure")
        self._log(query, decision)
        return decision

    def _log(self, query: str, decision: PolicyDecision) -> None:
        if self.audit_log_path is None:
            return
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).isoformat()
        with self.audit_log_path.open("a", encoding="utf-8") as fp:
            fp.write(
                f"{ts}\tallowed={decision.allowed}\trule_id={decision.rule_id}\t"
                f"reason={decision.reason}\tquery={query.strip()}\n"
            )

    def audit_summary(self) -> dict[str, float]:
        if self.audit_log_path is None or not self.audit_log_path.exists():
            return {"policy_events": 0.0, "blocked_events": 0.0, "blocked_ratio": 0.0}
        total = 0
        blocked = 0
        with self.audit_log_path.open("r", encoding="utf-8") as fp:
            for line in fp:
                text = line.strip()
                if not text:
                    continue
                total += 1
                if "allowed=False" in text:
                    blocked += 1
        ratio = (blocked / total) if total else 0.0
        return {"policy_events": float(total), "blocked_events": float(blocked), "blocked_ratio": ratio}

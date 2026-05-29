from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

PROMPT_INJECTION_PATTERNS = ("ignore previous", "system prompt", "jailbreak", "bypass safety")
PROHIBITED_ADVICE_PATTERNS = ("guaranteed return", "sure-shot", "all-in", "insider tip")


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    reason: str


class SafetyPolicy:
    def __init__(self, audit_log_path: Path | None):
        self.audit_log_path = audit_log_path

    def evaluate(self, query: str) -> PolicyDecision:
        q = query.lower()
        if any(p in q for p in PROMPT_INJECTION_PATTERNS):
            decision = PolicyDecision(False, "Prompt-injection pattern detected")
            self._log(query, decision)
            return decision
        if any(p in q for p in PROHIBITED_ADVICE_PATTERNS):
            decision = PolicyDecision(False, "Prohibited unsafe-advice pattern detected")
            self._log(query, decision)
            return decision
        decision = PolicyDecision(True, "Allowed with SEBI-aligned disclosure requirements")
        self._log(query, decision)
        return decision

    def _log(self, query: str, decision: PolicyDecision) -> None:
        if self.audit_log_path is None:
            return
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).isoformat()
        with self.audit_log_path.open("a", encoding="utf-8") as fp:
            fp.write(f"{ts}\tallowed={decision.allowed}\treason={decision.reason}\tquery={query.strip()}\n")


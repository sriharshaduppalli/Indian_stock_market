from pathlib import Path

from indian_stock_llm.acceptance import ProductionAcceptanceCriteria, SUPPORTED_QUERY_CATEGORIES
from indian_stock_llm.config import AssistantConfig
from indian_stock_llm.data_layer import EnterpriseDataLayer
from indian_stock_llm.evaluation import BenchmarkResult, passes_release_gate
from indian_stock_llm.query_engine import StockMarketAssistant
from indian_stock_llm.release_manager import ReleaseRegistry
from indian_stock_llm.serving import ChatService


def _config(tmp_path: Path) -> AssistantConfig:
    repo_root = Path(__file__).resolve().parents[1]
    return AssistantConfig(
        knowledge_base_path=repo_root / "data" / "sample_knowledge.json",
        instrument_master_path=repo_root / "data" / "enterprise" / "instrument_master.json",
        corporate_actions_path=repo_root / "data" / "enterprise" / "corporate_actions.json",
        filings_path=repo_root / "data" / "enterprise" / "filings.json",
        regulatory_updates_path=repo_root / "data" / "enterprise" / "regulatory_updates.json",
        market_events_path=repo_root / "data" / "enterprise" / "market_events.json",
        feedback_log_path=tmp_path / "feedback.log",
        policy_audit_log_path=tmp_path / "policy.log",
        release_registry_path=tmp_path / "release_registry.json",
    )


def test_supported_categories_defined() -> None:
    assert set(SUPPORTED_QUERY_CATEGORIES) == {
        "stocks",
        "nse_bse_sebi",
        "analysis",
        "calculations",
        "prediction_guidance",
    }


def test_enterprise_data_layer_snapshot_and_validation(tmp_path: Path) -> None:
    layer = EnterpriseDataLayer(_config(tmp_path))
    assert layer.validate_snapshot() is True
    assert layer.snapshot.source_hierarchy
    assert "instrument_master" in layer.snapshot.lineage


def test_evaluation_gate_thresholds() -> None:
    criteria = ProductionAcceptanceCriteria()
    strong = BenchmarkResult(0.9, 0.92, 0.88, 0.05, 0.99, 0.9)
    weak = BenchmarkResult(0.8, 0.7, 0.75, 0.2, 0.9, 0.8)
    assert passes_release_gate(strong, criteria) is True
    assert passes_release_gate(weak, criteria) is False


def test_chat_service_cache_and_contract(tmp_path: Path) -> None:
    assistant = StockMarketAssistant(config=_config(tmp_path))
    service = ChatService(assistant=assistant, rate_limit_per_minute=5)
    first = service.query("What is SEBI role in market regulation?")
    second = service.query("What is SEBI role in market regulation?")
    assert first["status"] == "ok"
    assert first["response"]["intent"]
    assert second.get("cached") is True


def test_release_registry_versioning_and_rollback(tmp_path: Path) -> None:
    config = _config(tmp_path)
    registry = ReleaseRegistry(config.release_registry_path)
    registry.add_version("v1", "baseline")
    registry.add_version("v2", "retrieval upgrade")
    assert registry.rollback_target() == "v1"


def test_chat_service_circuit_recovery_with_cooldown() -> None:
    class FailingAssistant:
        def __init__(self) -> None:
            self.calls = 0

        def query(self, _query: str) -> dict:
            self.calls += 1
            if self.calls <= 2:
                raise RuntimeError("temporary failure")
            return {
                "intent": "general_query",
                "answer": "ok",
                "confidence": 0.5,
                "citations": [],
                "disclaimer": "Informational only.",
                "safe_for_trading_advice": False,
            }

    service = ChatService(
        assistant=FailingAssistant(),
        rate_limit_per_minute=10,
        circuit_breaker_threshold=1,
        circuit_cooldown_seconds=0,
    )
    first = service.query("test failure")
    second = service.query("test recovery")
    assert first["status"] == "failed"
    assert second["status"] == "ok"

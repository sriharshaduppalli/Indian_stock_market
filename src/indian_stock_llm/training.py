"""LoRA fine-tuning infrastructure for the Indian Stock Market assistant.

Provides:
- ``QAPairBuilder``: converts knowledge-base items to instruction-tuning format.
- ``LoRAFineTuner``: wraps HuggingFace ``transformers`` + ``peft`` for LoRA fine-tuning.

Both classes degrade gracefully when optional dependencies are unavailable.

Requirements (optional):
    pip install transformers peft accelerate torch
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .knowledge_base import KnowledgeItem

_TAG_TO_INTENT: dict[str, str] = {
    "fundamentals": "fundamentals",
    "valuation": "fundamentals",
    "pe": "fundamentals",
    "sebi": "events_news",
    "regulation": "events_news",
    "earnings": "events_news",
    "analysis": "stock_analysis",
    "technical": "stock_analysis",
    "prediction": "prediction",
    "forecast": "prediction",
    "calculation": "market_calculations",
    "cagr": "market_calculations",
    "portfolio": "portfolio",
    "risk": "portfolio",
}

_INTENT_INSTRUCTIONS: dict[str, str] = {
    "fundamentals": (
        "Explain the fundamental valuation and financial health aspects "
        "of the following Indian stock market query."
    ),
    "events_news": (
        "Summarize the regulatory or corporate event context relevant "
        "to the following Indian market query."
    ),
    "stock_analysis": (
        "Provide a technical and fundamental analysis for the following Indian stock query."
    ),
    "prediction": (
        "Provide a probabilistic, risk-aware outlook for the following Indian stock market "
        "prediction query. Do not make guaranteed-return claims."
    ),
    "market_calculations": (
        "Perform or explain the financial calculation requested in the following query."
    ),
    "portfolio": (
        "Provide risk-aware portfolio guidance for the following Indian equities query."
    ),
    "general_query": (
        "Answer the following Indian stock market question using grounded, factual context."
    ),
}


@dataclass(frozen=True)
class QAPair:
    """A single instruction-tuning record."""

    instruction: str
    input: str
    output: str
    source: str = "knowledge_base"


class QAPairBuilder:
    """Builds instruction-tuning QA pairs from a KnowledgeBase item list."""

    def _intent_for(self, item: KnowledgeItem) -> str:
        for tag in item.tags:
            mapped = _TAG_TO_INTENT.get(tag.lower())
            if mapped:
                return mapped
        return "general_query"

    def from_knowledge_base(self, items: list[KnowledgeItem]) -> list[QAPair]:
        """Convert KnowledgeItem list to QAPair training records."""
        pairs: list[QAPair] = []
        for item in items:
            intent = self._intent_for(item)
            instruction = _INTENT_INSTRUCTIONS.get(intent, _INTENT_INSTRUCTIONS["general_query"])
            output = (
                f"{item.content}\n\n"
                f"Source: {item.source}. "
                "Validate with live NSE/BSE data before making investment decisions."
            )
            pairs.append(
                QAPair(
                    instruction=instruction,
                    input=item.title,
                    output=output,
                    source=item.source,
                )
            )
        return pairs

    def save(self, pairs: list[QAPair], path: Path) -> None:
        """Serialize QA pairs to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        records = [
            {
                "instruction": p.instruction,
                "input": p.input,
                "output": p.output,
                "source": p.source,
            }
            for p in pairs
        ]
        path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, path: Path) -> list[QAPair]:
        """Load QA pairs from a previously saved JSON file."""
        records = json.loads(path.read_text(encoding="utf-8"))
        return [QAPair(**r) for r in records]


class LoRAFineTuner:
    """LoRA fine-tuning wrapper using HuggingFace transformers + peft.

    Requires: pip install transformers peft accelerate torch
    (available via requirements-optional.txt)

    Falls back to a clear RuntimeError if dependencies are missing.
    """

    def __init__(
        self,
        base_model: str,
        output_path: Path,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_target_modules: tuple[str, ...] = ("q_proj", "v_proj"),
        lora_dropout: float = 0.1,
    ) -> None:
        self.base_model = base_model
        self.output_path = Path(output_path)
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_target_modules = lora_target_modules
        self.lora_dropout = lora_dropout

    def _require_dependencies(self):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments  # noqa: F401
            from peft import LoraConfig, TaskType, get_peft_model  # noqa: F401
            import torch  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "LoRA fine-tuning requires optional dependencies. "
                "Install them with: pip install transformers peft accelerate torch"
            ) from exc

    def train(
        self,
        pairs: list[QAPair],
        *,
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-4,
    ) -> Path:
        """Run LoRA fine-tuning on the provided QA pairs.

        Returns the output path where the adapter and tokenizer are saved.
        """
        self._require_dependencies()

        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
        from peft import LoraConfig, TaskType, get_peft_model
        import torch

        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(self.base_model)
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            target_modules=list(self.lora_target_modules),
            lora_dropout=self.lora_dropout,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        texts = [
            f"### Instruction:\n{p.instruction}\n\n### Input:\n{p.input}\n\n### Response:\n{p.output}"
            for p in pairs
        ]
        encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt",
        )

        class _Dataset(torch.utils.data.Dataset):
            def __init__(self, enc) -> None:
                self.enc = enc

            def __len__(self) -> int:
                return len(self.enc["input_ids"])

            def __getitem__(self, idx: int) -> dict:
                return {k: v[idx] for k, v in self.enc.items()}

        dataset = _Dataset(encodings)
        training_args = TrainingArguments(
            output_dir=str(self.output_path),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            save_strategy="epoch",
            logging_steps=10,
            report_to="none",
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
        )
        trainer.train()
        model.save_pretrained(str(self.output_path))
        tokenizer.save_pretrained(str(self.output_path))
        return self.output_path

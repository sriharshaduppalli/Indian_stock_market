from __future__ import annotations

import argparse
import json
import sys

from .query_engine import StockMarketAssistant


def main() -> None:
    args_list = sys.argv[1:]
    # If first positional arg is a known subcommand, dispatch to it.
    # Otherwise fall back to legacy direct-query mode for full backward compatibility.
    if args_list and args_list[0] in ("benchmark", "train"):
        _dispatch_subcommand(args_list)
        return

    # Legacy mode: direct query
    parser = argparse.ArgumentParser(description="Indian Stock Market LLM Assistant (scaffold)")
    parser.add_argument("query", help="User query about Indian stocks")
    parser.add_argument("--json", action="store_true", help="Return API-friendly JSON response")
    args = parser.parse_args(args_list)

    assistant = StockMarketAssistant()
    if args.json:
        print(json.dumps(assistant.query(args.query), ensure_ascii=False, indent=2))
        return
    response = assistant.ask(args.query)
    print(response.answer)


def _dispatch_subcommand(args_list: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Indian Stock Market LLM Assistant — subcommands")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("benchmark", help="Run the built-in benchmark suite").add_argument(
        "--json", action="store_true", help="Return JSON benchmark report"
    )

    train_parser = subparsers.add_parser("train", help="Build QA pairs from the knowledge base")
    train_parser.add_argument("--output", required=True, help="Output path for QA pairs JSON")
    train_parser.add_argument(
        "--finetune", action="store_true", help="Run LoRA fine-tuning after building QA pairs"
    )
    train_parser.add_argument("--base-model", help="HuggingFace model ID for fine-tuning")
    train_parser.add_argument("--lora-output", help="Output directory for LoRA adapter weights")

    args = parser.parse_args(args_list)

    if args.command == "benchmark":
        _run_benchmark(json_output=getattr(args, "json", False))
    elif args.command == "train":
        _run_train(args)


def _run_benchmark(json_output: bool) -> None:
    from .evaluation import BenchmarkSuite

    assistant = StockMarketAssistant()
    suite = BenchmarkSuite()
    result, details = suite.run(assistant)
    if json_output:
        print(
            json.dumps(
                {
                    "routing_accuracy": result.routing_accuracy,
                    "fact_accuracy": result.fact_accuracy,
                    "groundedness": result.groundedness,
                    "hallucination_rate": result.hallucination_rate,
                    "safety_score": result.safety_score,
                    "calculation_correctness": result.calculation_correctness,
                    "details": details,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    print("Benchmark Results:")
    print(f"  Routing accuracy:        {result.routing_accuracy:.1%}")
    print(f"  Fact accuracy:           {result.fact_accuracy:.1%}")
    print(f"  Groundedness:            {result.groundedness:.1%}")
    print(f"  Hallucination rate:      {result.hallucination_rate:.1%}")
    print(f"  Safety score:            {result.safety_score:.1%}")
    print(f"  Calculation correctness: {result.calculation_correctness:.1%}")


def _run_train(args) -> None:
    from pathlib import Path

    from .training import LoRAFineTuner, QAPairBuilder

    assistant = StockMarketAssistant()
    builder = QAPairBuilder()
    pairs = builder.from_knowledge_base(assistant.knowledge_base.items)
    output_path = Path(args.output)
    builder.save(pairs, output_path)
    print(f"Built {len(pairs)} QA pairs → {output_path}")

    if args.finetune:
        if not args.base_model:
            print("Error: --base-model is required for fine-tuning")
            return
        lora_output = Path(args.lora_output) if args.lora_output else output_path.parent / "lora_adapter"
        finetuner = LoRAFineTuner(base_model=args.base_model, output_path=lora_output)
        print(f"Starting LoRA fine-tuning of {args.base_model} …")
        result_path = finetuner.train(pairs)
        print(f"LoRA adapter saved to {result_path}")


if __name__ == "__main__":
    main()

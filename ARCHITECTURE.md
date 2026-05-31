# Indian Stock Market LLM Architecture

## 1) Product Goal
Build a domain-focused assistant for Indian equities that answers user queries with high factual grounding, low hallucination risk, and clear uncertainty handling.

## 2) Core Principles
- **Domain-first**: NSE/BSE entities, Indian regulations, sector context, and India-specific market microstructure.
- **Grounded responses**: Retrieval-augmented generation from trusted datasets.
- **Multi-stage reasoning**: Intent detection → retrieval → analysis → answer generation.
- **Safety**: No guaranteed-return claims; include risk disclosures.

## 3) System Components
1. **Data Ingestion Layer**
   - Market data: OHLCV, corporate actions, filings, indices.
   - Text data: annual reports, earnings call transcripts, news, circulars.
   - Live streams: breaking news feeds, macro calendar, sentiment/event updates.
   - Normalize to common schema and timestamp.
2. **Feature + Knowledge Layer**
   - Structured features: valuation ratios, momentum, volatility, drawdown, quality factors.
   - Predictive factors: news sentiment, event surprise score, sector-relative strength, liquidity regime.
   - Unstructured store: chunked documents with metadata.
3. **Retrieval Layer (RAG)**
   - Hybrid retrieval: keyword + semantic retrieval with n-gram Jaccard scoring.
   - `LocalHashEmbeddingProvider` (default, zero-dep) or `SentenceTransformerEmbeddingProvider` for real semantic embeddings (optional, `sentence-transformers`).
   - `HeuristicReranker` (default) or `MLReranker` (logistic regression on self-supervised features, persists weights as JSON; requires `scikit-learn` for training only).
   - `HttpReranker` for managed embedding/reranker endpoints.
   - `KnowledgeBase.refresh_index()` rebuilds embeddings in-place; called automatically by the nightly refresh daemon.
4. **Model Layer**
   - Base LLM (instruction-tuned) via `HttpModelBackend` or `TemplateModelBackend` (stub).
   - Optional LoRA fine-tuning using Indian-market QA pairs via `LoRAFineTuner` (requires `transformers`, `peft`).
   - `QAPairBuilder` converts knowledge-base items to instruction-tuning records (JSONL).
5. **Prediction Layer**
   - `PredictionEngine` scores context items for bullish/bearish signals and generates multi-horizon predictions: intraday, swing (1–5 days), medium-term (1–3 months).
   - Each horizon returns `direction`, `probability`, and `rationale`.
   - Signals are serialized as `prediction_signals` in the `query()` response dict when intent is `"prediction"`.
   - Max probability capped at 0.75; never claims certainty.
6. **Inference Orchestrator** (`StockMarketAssistant`)
   - Query intent classifier → tool routing → retrieval → reranking → response composition.
   - Wires embedding provider, reranker, feedback analyzer, prediction engine, and nightly refresh daemon from `AssistantConfig`.
   - `trigger_index_refresh()` for on-demand index rebuilds.
   - `_start_nightly_refresh(hour_utc)` daemon thread for scheduled nightly rebuilds.
7. **Continual Learning Layer**
   - Daily feedback ingestion from user interactions and realized outcomes (`ContinualLearningManager`).
   - `DailyFeedbackAnalyzer` parses TSV and JSON-line log formats; returns intent distribution and readiness flag (≥10 samples).
   - `suggested_knowledge_refresh_tags()` surfaces which knowledge topics to prioritize based on recent query patterns.
8. **Evaluation Layer**
   - `BenchmarkSuite` with 9 domain seed cases (fundamentals, prediction, events, calculations, safety).
   - Metrics: `routing_accuracy`, `fact_accuracy`, `groundedness`, `hallucination_rate`, `safety_score`, `calculation_correctness`.
   - `evaluate_release_gate()` gates production deployment on benchmark thresholds.
9. **Low-Latency Serving Layer**
   - `ChatService` with circuit-breaker, rate-limit, cache, SLO alerts, and tenant-scoped auth.
   - `ChatService.refresh()` triggers an immediate knowledge-base index rebuild.
   - `ChatApi.refresh()` exposes refresh via the stable v1 API contract.
   - `POST /admin/refresh` HTTP endpoint (admin token gated) for ops-triggered rebuilds.
   - `GET /health`, `GET /metrics`, `POST /query` existing routes unchanged.

## 4) Training and Daily Improvement Loop
1. Collect domain corpus, live-news features, and QA seeds.
2. Train retrieval index + baseline assistant + prediction head.
3. Fine-tune with instruction data via `QAPairBuilder` → `LoRAFineTuner` (LoRA adapters).
4. Run daily incremental update jobs (index refresh via `trigger_index_refresh`, factor recalibration, drift checks).
5. Evaluate on `BenchmarkSuite`; run red-team prompts; check calibration metrics.
6. Deploy with latency SLO monitoring and capture feedback for next-day learning.

## 5) CLI Subcommands
```bash
# Legacy direct query (backward compatible)
python -m indian_stock_llm.cli "What are valuation risks in Indian IT stocks?"

# Run the benchmark suite
python -m indian_stock_llm.cli benchmark
python -m indian_stock_llm.cli benchmark --json

# Build QA pairs for fine-tuning
python -m indian_stock_llm.cli train --output qa_pairs.json

# Build QA pairs and run LoRA fine-tuning
python -m indian_stock_llm.cli train --output qa_pairs.json --finetune \
    --base-model google/flan-t5-base --lora-output ./adapter
```

## 6) Configuration (ISM_* env vars)
| Variable | Purpose |
|---|---|
| `ISM_EMBEDDING_LOCAL_MODEL` | SentenceTransformer model name (e.g. `all-MiniLM-L6-v2`) |
| `ISM_NIGHTLY_REFRESH_ENABLED` | Enable nightly index refresh daemon (`true`/`false`) |
| `ISM_NIGHTLY_REFRESH_HOUR_UTC` | UTC hour for nightly refresh (integer) |
| `ISM_TRAINING_BASE_MODEL` | HuggingFace base model for LoRA fine-tuning |
| `ISM_TRAINING_LORA_RANK` | LoRA rank (integer, default 8) |
| `ISM_TRAINING_LORA_ALPHA` | LoRA alpha (integer, default 16) |
| `ISM_TRAINING_DATA_PATH` | Output path for QA pairs JSON |
| `ISM_TRAINING_OUTPUT_PATH` | Output path for LoRA adapter weights |
| `ISM_RERANKER_LOCAL_MODEL_PATH` | Path to persisted MLReranker weights JSON |

## 7) Initial Implementation Scope in this Repository
- A lightweight architecture scaffold with:
  - Configurable knowledge base with real semantic embeddings (optional).
  - Intent classification with 8 domain intents.
  - Hybrid retrieval over local domain documents with ML-based reranking (optional).
  - Response composer with citations and safety disclaimer.
  - Multi-horizon prediction signals for prediction-intent queries.
  - LoRA fine-tuning infrastructure for domain adaptation.
  - Benchmark suite with 9 domain seed cases.
  - Nightly index refresh daemon and `/admin/refresh` endpoint.
  - Continual-learning hook with intent distribution analysis.
- This is the foundation to plug in a production LLM and real-time market connectors.

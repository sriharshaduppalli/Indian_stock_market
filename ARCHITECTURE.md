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
   - Hybrid retrieval: keyword + semantic retrieval.
   - Re-ranker for domain relevance (ticker, sector, time window).
4. **Model Layer**
   - Base LLM (instruction-tuned) + domain adaptation.
   - Optional LoRA fine-tuning using Indian-market QA pairs.
5. **Prediction Layer**
   - Multi-horizon prediction head (intraday / swing / medium-term probabilities).
   - Combine time-series signals + live-news impact features.
   - Calibrated confidence and uncertainty output.
6. **Inference Orchestrator**
   - Query intent classifier.
   - Tool routing (quote lookup, fundamentals, event/news, portfolio what-if, prediction mode).
   - Prompt builder with explicit context citations.
7. **Continual Learning Layer**
   - Daily feedback ingestion from user interactions and realized outcomes.
   - Nightly refresh of retrieval index and factor weights.
   - Scheduled fine-tuning cycles with drift detection triggers.
8. **Evaluation Layer**
   - Factuality and citation correctness.
   - Domain benchmark set (earnings, ratios, corporate actions, sector trends).
   - Response quality metrics: relevance, completeness, risk language, prediction calibration.
9. **Low-Latency Serving Layer**
   - Distilled/quantized fast model for online inference.
   - Caching for repeated queries and hot tickers.
   - Async retrieval + precomputed features to keep p95 latency low.

## 4) Training and Daily Improvement Loop
1. Collect domain corpus, live-news features, and QA seeds.
2. Train retrieval index + baseline assistant + prediction head.
3. Fine-tune with instruction data, preference pairs, and realized market outcomes.
4. Run daily incremental update jobs (index refresh, factor recalibration, drift checks).
5. Evaluate on benchmark + calibration metrics; run red-team prompts.
6. Deploy with latency SLO monitoring and capture feedback for next-day learning.

## 5) Initial Implementation Scope in this Repository
- A lightweight architecture scaffold with:
  - Configurable knowledge base.
  - Intent classification.
  - Retrieval over local domain documents.
  - Response composer with citations and safety disclaimer.
  - Prediction intent response path with live-impact factor checklist.
  - Continual-learning hook for daily feedback logs.
- This is the foundation to plug in a production LLM and real-time market connectors.

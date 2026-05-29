# Indian_stock_market

Indian stock market data, analysis, prompts, queries, and LLM model scaffold.

## What is implemented
- High-level architecture in `ARCHITECTURE.md`
- Initial local assistant scaffold for Indian stock market Q&A:
  - Intent classification
    - Covers general Indian stock market, NSE/BSE/SEBI context, stock analysis, market calculations, and prediction guidance intents
  - Domain knowledge retrieval
    - Hybrid retrieval (keyword + semantic overlap) with metadata filtering
    - ISIN/company-level symbol/entity resolution via enterprise instrument master
  - Grounded response composition with source citations
  - Production-style response metadata (confidence, citations, disclaimer, safety flag)
  - Prediction-intent path with live-impact factor guidance
  - Deterministic calculation support for common return/CAGR prompts
  - SEBI-aligned safety checks, prompt-injection defenses, and policy audit logs
  - Enterprise data layer scaffold with source hierarchy, validation, refresh, and lineage metadata
  - Android-ready chat serving scaffold (stable contract, cache, rate-limit, retries, circuit-breaker, fallback)
  - Evaluation release-gate scaffold for factuality, groundedness, hallucination, routing, and safety metrics
  - Versioned release registry + rollback target helper
  - Daily continual-learning feedback hook
  - Fast latency mode configuration

## Quick start
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run tests:
   ```bash
   PYTHONPATH=src pytest -q
   ```
3. Ask a query:
   ```bash
   PYTHONPATH=src python -m indian_stock_llm.cli "What are valuation risks in Indian IT stocks?"
   ```
4. API-friendly JSON output:
   ```bash
   PYTHONPATH=src python -m indian_stock_llm.cli --json "Predict NIFTY next week"
   ```

## Next steps for production-grade accuracy
- Replace scaffold feeds with live NSE/BSE + filings/news connectors
- Upgrade semantic retrieval to embedding + reranking stack
- Fine-tune base LLM with broader Indian market supervision data
- Integrate offline+online evaluation for continuous improvement rollouts
- Connect serving metrics/traces to production monitoring backends

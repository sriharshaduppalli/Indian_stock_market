# Indian_stock_market

Indian stock market data, analysis, prompts, queries, and LLM model scaffold.

## What is implemented
- High-level architecture in `ARCHITECTURE.md`
- Initial local assistant scaffold for Indian stock market Q&A:
  - Intent classification
    - Covers general Indian stock market, NSE/BSE/SEBI context, stock analysis, market calculations, and prediction intents
  - Domain knowledge retrieval
  - Grounded response composition with source citations
  - Production-style response metadata (confidence, citations, disclaimer, safety flag)
  - Prediction-intent path with live-impact factor guidance
  - Deterministic calculation support for common return/CAGR prompts
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
- Integrate live NSE/BSE + filings/news pipelines
- Add vector retrieval + reranking
- Fine-tune a base LLM with Indian market instruction data
- Add robust evaluation benchmark and guardrails
- Add automated daily retraining jobs and latency SLO dashboards

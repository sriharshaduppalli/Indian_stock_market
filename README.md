# Indian_stock_market

Indian stock market data, analysis, prompts, queries, and LLM model scaffold.

## What is implemented
- High-level architecture in `/tmp/workspace/sriharshaduppalli/Indian_stock_market/ARCHITECTURE.md`
- Initial local assistant scaffold for Indian stock market Q&A:
  - Intent classification
  - Domain knowledge retrieval
  - Grounded response composition with source citations

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

## Next steps for production-grade accuracy
- Integrate live NSE/BSE + filings/news pipelines
- Add vector retrieval + reranking
- Fine-tune a base LLM with Indian market instruction data
- Add robust evaluation benchmark and guardrails

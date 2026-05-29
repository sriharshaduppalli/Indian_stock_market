from __future__ import annotations

import argparse

from .query_engine import StockMarketAssistant


def main() -> None:
    parser = argparse.ArgumentParser(description="Indian Stock Market LLM Assistant (scaffold)")
    parser.add_argument("query", help="User query about Indian stocks")
    args = parser.parse_args()

    assistant = StockMarketAssistant()
    response = assistant.ask(args.query)
    print(response.answer)


if __name__ == "__main__":
    main()

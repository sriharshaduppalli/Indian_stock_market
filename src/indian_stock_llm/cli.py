from __future__ import annotations

import argparse
import json

from .query_engine import StockMarketAssistant


def main() -> None:
    parser = argparse.ArgumentParser(description="Indian Stock Market LLM Assistant (scaffold)")
    parser.add_argument("query", help="User query about Indian stocks")
    parser.add_argument("--json", action="store_true", help="Return API-friendly JSON response")
    args = parser.parse_args()

    assistant = StockMarketAssistant()
    if args.json:
        print(json.dumps(assistant.query(args.query), ensure_ascii=False, indent=2))
        return

    response = assistant.ask(args.query)
    print(response.answer)


if __name__ == "__main__":
    main()

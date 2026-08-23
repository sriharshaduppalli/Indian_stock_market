"""Rich NSE official-market literacy for the Indian Stock LLM RAG index.

This is curated education mapped to what https://www.nseindia.com/ publishes
(quotes, chains, holidays, circulars, Learn). It is NOT a scrape, crawl, or
fine-tune of NSE HTML. Live lots, holidays, circuits, and circular numbers
must be confirmed on the exchange site — never invented here.
"""

from __future__ import annotations

from .knowledge_base import KnowledgeItem

_SRC = "bysel_nse_literacy_v2"


def nse_official_literacy_items() -> list[KnowledgeItem]:
    raw = [
        {
            "id": "nse_official_source",
            "title": "NSE official website is the rulebook",
            "content": (
                "The National Stock Exchange of India (NSE) is the official venue for most "
                "Indian equity cash and index/stock F&O that BYSEL discusses in paper practice. "
                "Listings, lot sizes, expiries, circuit bands, holidays, circulars, and index "
                "methodology change. BYSEL does not crawl or fine-tune on "
                "https://www.nseindia.com/. We teach the map. For any live rule, holiday, lot, "
                "or circular: open nseindia.com and confirm. Never invent a circular number, "
                "lot size, expiry date, or holiday."
            ),
            "tags": [
                "nse", "nseindia", "official", "circular", "holiday", "lot size",
                "education", "website",
            ],
            "source": _SRC,
        },
        {
            "id": "nse_site_map",
            "title": "nseindia.com information map",
            "content": (
                "Typical official pages a learner uses on https://www.nseindia.com/: "
                "Market Data / Get Quote (LTP, OHLC, volume, 52-week, corporate actions); "
                "Live option chain (strikes, OI, IV, PCR-style tape); "
                "Market status / timings; Holidays calendar; "
                "Circulars and regulations (lot-size, settlement, product changes); "
                "Indices (Nifty family factsheets); "
                "Learn / Education / NCFM (investor awareness, not tips); "
                "IPO / listed products. BYSEL paper quotes are a practice tape — "
                "the exchange page wins when you need an official figure."
            ),
            "tags": [
                "nseindia", "quote", "option chain", "holiday", "circular",
                "ncfm", "learn", "indices", "nse",
            ],
            "source": _SRC,
        },
        {
            "id": "nse_segments",
            "title": "NSE market segments",
            "content": (
                "NSE runs several segments. Capital Market (CM / equity cash): delivery and "
                "intraday in listed shares, ETFs, and some other cash products. "
                "Futures & Options (FO): index and stock derivatives with lots, expiry, "
                "and margin. Currency derivatives (CD) and other products have their own "
                "timings and specs. A name can trade in cash only, or cash plus F&O. "
                "Confirm product eligibility and contract specs on the NSE quote / contract "
                "info page — do not assume every Nifty name has weekly options."
            ),
            "tags": [
                "nse", "segment", "equity", "fno", "currency", "etf", "cash", "education",
            ],
            "source": _SRC,
        },
        {
            "id": "nse_session_clock",
            "title": "NSE session clock (verify circulars)",
            "content": (
                "Equity cash typically: pre-open around 9:00–9:15 IST, continuous from 9:15. "
                "From 3 Aug 2026 the close is multi-window: F&O-eligible cash continuous "
                "often ends ~3:15 then Closing Auction Session (CAS) into ~3:35 (close from "
                "auction); non-F&O cash often continuous to ~3:30; equity derivatives often "
                "to ~3:40; a short post-close window may exist. There is no longer one "
                "universal 3:30 close for every book. Broker MIS square-off for CAS names "
                "can differ. Always confirm current timings and special sessions on "
                "nseindia.com market status / circulars."
            ),
            "tags": [
                "market hours", "timings", "pre-open", "cas", "session", "nse", "ist",
            ],
            "source": _SRC,
        },
        {
            "id": "nse_get_quote",
            "title": "How to read an NSE Get Quote page",
            "content": (
                "On NSE Get Quote, literacy fields usually include: last traded price, "
                "open/high/low/previous close, volume and value, VWAP, 52-week high/low, "
                "circuit / price band, series (EQ vs others), ISIN, face value, and "
                "corporate-action dates. Bid/ask and trade info show the book tightness. "
                "A freeze or band means orders outside the band will not match. "
                "BYSEL cards show a practice last and OHLC — if band, series, or ISIN "
                "matters, open the official quote."
            ),
            "tags": [
                "quote", "ltp", "vwap", "52 week", "price band", "isin", "nseindia", "nse",
            ],
            "source": _SRC,
        },
        {
            "id": "nse_indices_family",
            "title": "NSE index family (Nifty)",
            "content": (
                "Nifty 50 is the flagship large-cap benchmark; Nifty Next 50, Midcap, "
                "Smallcap, Bank Nifty, Nifty IT, and sector/thematic indices also exist. "
                "Methodology is typically free-float market-cap with periodic reconstitution. "
                "Index futures/options use the official lot and expiry on the FO segment. "
                "Do not invent constituent weights or the current lot. Factsheets and "
                "methodology PDFs live on nseindia.com → Indices. BYSEL may show NIFTY50 / "
                "BANKNIFTY as practice symbols — official close and composition are on NSE."
            ),
            "tags": [
                "nifty", "banknifty", "index", "indices", "nifty 50", "benchmark", "nse",
            ],
            "source": _SRC,
        },
        {
            "id": "nse_circuits",
            "title": "Stock price bands vs index circuit breakers",
            "content": (
                "Individual stocks have price bands (circuits) that block trades beyond a "
                "percent from the reference price — often 2/5/10/20% by surveillance category; "
                "the exact band is on the quote. Index circuit breakers can halt the whole "
                "market on extreme Nifty/Sensex moves (multi-stage, time-of-day rules). "
                "A stock at upper circuit can be unbuyable; at lower circuit unsellable in "
                "the cash book. Never quote a band percent from memory — read NSE quote / "
                "circulars. BYSEL paper fills do not simulate every freeze."
            ),
            "tags": [
                "circuit", "price band", "freeze", "index circuit", "surveillance", "nse",
            ],
            "source": _SRC,
        },
        {
            "id": "nse_settlement_t1",
            "title": "T+1 settlement, delivery, auction",
            "content": (
                "Indian equity cash currently settles on T+1: buy today, shares/funds "
                "generally settle next trading day in demat/bank via the clearing corp. "
                "Intraday/MIS is squared the same session and does not take delivery. "
                "Short delivery can go to auction — that is different from the closing "
                "auction (CAS). Holidays shift the settlement calendar. Confirm current "
                "settlement and auction rules on NSE circulars. BYSEL paper delivery is a "
                "simulation, not a CDSL/NSDL credit."
            ),
            "tags": [
                "t+1", "settlement", "delivery", "auction", "demat", "cas", "nse",
            ],
            "source": _SRC,
        },
        {
            "id": "nse_fo_contracts",
            "title": "NSE F&O contracts: lot, expiry, margin",
            "content": (
                "F&O trades in whole lots. Lot size, freeze quantity, tick, and expiry "
                "(weekly and/or monthly depending on the underlier) are exchange specs and "
                "get revised by circular. Margin is SPAN + exposure style at the broker; "
                "MTM can wipe a book even if the longer view is intact. Stock F&O may have "
                "physical settlement near expiry — confirm the current rule on NSE. "
                "Never invent a Nifty or Bank Nifty lot. Open contract info / circulars on "
                "nseindia.com. BYSEL F&O screens are paper drills."
            ),
            "tags": [
                "fno", "lot size", "expiry", "margin", "span", "weekly", "physical settlement",
                "nse",
            ],
            "source": _SRC,
        },
        {
            "id": "nse_option_chain",
            "title": "How to read the NSE option chain",
            "content": (
                "The official option chain lists calls and puts by strike for a chosen expiry: "
                "LTP, volume, open interest (OI), change in OI, IV, and greeks if shown. "
                "OI is outstanding contracts, not the same as volume. Rising price + rising OI "
                "often means new positions; falling OI can mean covering/liquidation. "
                "PCR (put OI / call OI) is a crowding cue, not a buy signal. "
                "Max-pain style readings are educational folklore unless you compute them "
                "yourself from the official chain. Use nseindia.com option-chain for the live "
                "grid. BYSEL may show a synthetic chain when NSE is blocked — label it as such."
            ),
            "tags": [
                "option chain", "oi", "iv", "pcr", "max pain", "greeks", "options", "nseindia",
            ],
            "source": _SRC,
        },
        {
            "id": "nse_order_types",
            "title": "NSE-style order types (literacy)",
            "content": (
                "Common cash/F&O order types at brokers mapping to the exchange: "
                "Market (match now at available prices), Limit (price cap/floor), "
                "Stop-loss / trigger (becomes live after a trigger), and validity "
                "(DAY, IOC, etc.). GTT is usually a broker instruction, not an exchange "
                "order type. A price-band freeze rejects orders outside the band. "
                "Paper practice: write the type, trigger, and invalidation before tapping Buy. "
                "Confirm current allowed types and tick sizes on NSE / your broker."
            ),
            "tags": [
                "order", "limit", "market", "stop loss", "gtt", "ioc", "nse", "education",
            ],
            "source": _SRC,
        },
        {
            "id": "nse_corporate_actions",
            "title": "Corporate actions on the NSE quote",
            "content": (
                "Dividends, bonus, splits, rights, and buybacks appear on the company quote "
                "and corporate-actions area. Ex-date / record date decide entitlement — "
                "buying on or after ex-date usually misses that dividend/bonus. "
                "Splits and bonuses change share count and price; charts may adjust. "
                "Do not invent an ex-date. Read the official corporate-action line on "
                "nseindia.com. BYSEL quality/fundamentals never fabricate missing dates."
            ),
            "tags": [
                "dividend", "bonus", "split", "rights", "ex date", "corporate actions", "nse",
            ],
            "source": _SRC,
        },
        {
            "id": "nse_ipo_asba",
            "title": "IPO / ASBA on the official path",
            "content": (
                "Mainboard and SME IPOs are applied via brokers/UPI with ASBA: money is "
                "blocked and debited only if allotted. Allotment, listing date, and price "
                "band are official disclosures — not a BYSEL prediction. Listing happens "
                "on NSE and/or BSE as stated in the RHP. Grey-market premiums are unofficial "
                "and not an NSE figure. Confirm IPO calendar and documents on nseindia.com "
                "and SEBI filings."
            ),
            "tags": ["ipo", "asba", "listing", "rhp", "sme", "nse", "education"],
            "source": _SRC,
        },
        {
            "id": "nse_holidays",
            "title": "NSE holidays and special sessions",
            "content": (
                "Trading holidays (Republic Day, Holi, Diwali Laxmi Pujan, etc.) and "
                "Muhurat or special sessions are published on the NSE holidays / circulars "
                "pages and change by year. BYSEL may keep a session-open holiday set for "
                "clocks — if it disagrees with nseindia.com, the exchange wins. "
                "Never recite a full holiday list from chat memory as official."
            ),
            "tags": ["holiday", "muhurat", "special session", "calendar", "nse", "nseindia"],
            "source": _SRC,
        },
        {
            "id": "nse_fii_dii",
            "title": "FII/DII figures published via the exchange path",
            "content": (
                "Foreign and domestic institutional buy/sell aggregates are published on "
                "a regular schedule (often daily provisional). They are context for the "
                "tape, not a standalone buy/sell reason. Confirm the latest table on the "
                "official NSE / NSDL-style disclosures the site links — do not invent crores. "
                "Rupee, global risk, and earnings can dominate a single day’s flow print."
            ),
            "tags": ["fii", "dii", "fpi", "flows", "nse", "education"],
            "source": _SRC,
        },
        {
            "id": "nse_surveillance",
            "title": "ASM / GSM and surveillance (concept)",
            "content": (
                "NSE/SEBI surveillance can place names in additional or graded surveillance "
                "(ASM/GSM) or other lists: extra margins, tighter bands, or trade-for-trade. "
                "That is risk control, not a ‘buy the ban’ signal. The official list and "
                "reasons change — read circulars / surveillance on nseindia.com. "
                "BYSEL must not invent that a stock is in ASM."
            ),
            "tags": ["asm", "gsm", "surveillance", "trade for trade", "margin", "nse"],
            "source": _SRC,
        },
        {
            "id": "nse_bulk_block",
            "title": "Bulk and block deals (literacy)",
            "content": (
                "Bulk and block deals are large negotiated or reported trades disclosed "
                "on the exchange. They can hint at institutional activity but are not a "
                "tip. Quantity, price, and client category appear on official deal pages. "
                "Do not treat one block as insider knowledge. Confirm on nseindia.com "
                "market data → bulk/block when the user asks."
            ),
            "tags": ["bulk deal", "block deal", "institutional", "nse", "education"],
            "source": _SRC,
        },
        {
            "id": "nse_etf_sgb",
            "title": "ETFs and listed products on NSE",
            "content": (
                "Index ETFs, gold ETFs, and other ETFs trade in the cash book like shares "
                "with an iNAV / premium-discount to underlying. Sovereign Gold Bonds and "
                "some debt products have their own quote pages. Liquidity (volume, spread) "
                "matters more than the label ‘ETF’. Confirm AUM, tracking, and series on "
                "the official quote — BYSEL paper ETF/SGB screens are educational."
            ),
            "tags": ["etf", "sgb", "gold", "inav", "nse", "education"],
            "source": _SRC,
        },
        {
            "id": "nse_learn_ncfm",
            "title": "NSE Learn / NCFM is literacy, not a system",
            "content": (
                "NSE Learn, investor-awareness modules, and NCFM certifications teach "
                "products, risks, settlement, and ethics. That is education. It is not a "
                "secret strategy, not a SEBI research report, and not a buy/sell call. "
                "If the user asks for ‘NSE strategies’, teach process: know the product, "
                "read the official quote/chain, size so one stop cannot wipe the paper book, "
                "journal the plan. Then point to nseindia.com Learn / circulars for official "
                "wording and current fees/modules."
            ),
            "tags": [
                "ncfm", "learn", "strategy", "education", "nseindia", "tips", "sebi",
            ],
            "source": _SRC,
        },
        {
            "id": "nse_investor_grievance",
            "title": "Investor grievance path (NSE / SEBI)",
            "content": (
                "For broker or listed-company issues, official paths include the exchange "
                "investor grievance mechanism and SEBI SCORES. Keep contract notes, "
                "unique client codes, and timestamps. BYSEL is not a broker and cannot "
                "file a complaint for you. We do not invent ticket IDs. Point the user to "
                "nseindia.com investor / SEBI SCORES for the live form."
            ),
            "tags": ["grievance", "scores", "sebi", "investor", "complaint", "nse"],
            "source": _SRC,
        },
        {
            "id": "nse_rules_always_verify",
            "title": "Lot size, expiry, circuits — verify on NSE",
            "content": (
                "Lots, weekly/monthly expiries, price bands, and index breaker stages are "
                "revised. A number in chat or an old PDF can be stale. When asked for a lot, "
                "expiry, holiday, or circuit percent: explain the concept, then tell the user "
                "to confirm on nseindia.com quote / contract info / circulars / holidays. "
                "Do not guess 25/50/75 or a holiday list."
            ),
            "tags": [
                "lot size", "expiry", "circuit", "holiday", "verification", "fno", "nse",
            ],
            "source": _SRC,
        },
        {
            "id": "nse_practice_process",
            "title": "Official-style paper practice process",
            "content": (
                "Aligned with exchange investor-awareness themes, not a trading system: "
                "(1) Know the product — cash delivery vs intraday vs F&O — before sizing. "
                "(2) Read the official quote/chain so lot, freeze, and OI are not invented. "
                "(3) Size so one stop cannot ruin the paper book. "
                "(4) Journal plan and invalidation; do not chase social-media tips. "
                "(5) Live orders need a SEBI-registered broker. No guaranteed return. "
                "No ‘NSE secret strategy’."
            ),
            "tags": [
                "strategy", "process", "paper", "practice", "risk", "journaling", "nse",
            ],
            "source": _SRC,
        },
        {
            "id": "nse_investor_education_not_tips",
            "title": "NSE education vs trading tips",
            "content": (
                "Official investor education covers what equity, F&O, and currency are, "
                "how margin and settlement work, and why leverage can wipe a book. "
                "Turning ‘NSE Learn’ into ‘buy this now’ is a misuse. BYSEL answers stay "
                "educational. For live wording, send the user to nseindia.com Learn / circulars."
            ),
            "tags": ["nse", "nseindia", "strategy", "learn", "tips", "education", "sebi"],
            "source": _SRC,
        },
        {
            "id": "nse_currency_cd",
            "title": "NSE currency derivatives (high level)",
            "content": (
                "Currency futures/options (e.g. USDINR) trade in the CD segment with their "
                "own lot, tick, and session hours — not the same clock as equity cash. "
                "Pricing literacy includes interest-rate parity style carry, not a FX tip. "
                "Confirm contract specs and timings on nseindia.com currency pages / circulars."
            ),
            "tags": ["currency", "usdinr", "cd", "derivatives", "nse", "education"],
            "source": _SRC,
        },
        {
            "id": "nse_bysel_boundary",
            "title": "What BYSEL uses from NSE vs what it does not",
            "content": (
                "BYSEL may use public NSE-style endpoints already in the app (equity quote, "
                "shareholding, option-chain when reachable, equity master CSV) for paper "
                "practice and quality screens. If those calls fail, we show unavailable or "
                "a labelled synthetic — we do not scrape Learn pages or train weights on "
                "the website. Chat answers use this literacy pack plus live enrich. "
                "Official close, lot, and circular text still live on nseindia.com."
            ),
            "tags": [
                "bysel", "nseindia", "scrape", "data", "option chain", "shareholding",
                "education",
            ],
            "source": _SRC,
        },
    ]
    return [KnowledgeItem(**item) for item in raw]

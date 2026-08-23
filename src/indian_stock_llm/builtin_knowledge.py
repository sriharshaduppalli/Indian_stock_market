"""Production Indian-market knowledge pack for BYSEL's local LLM tier.

These items are always merged into the RAG index so educational, equation,
market-mechanics, and analysis queries stay grounded without a remote model.
"""

from __future__ import annotations

from .knowledge_base import KnowledgeItem
from .nse_official_literacy import nse_official_literacy_items


def builtin_knowledge_items() -> list[KnowledgeItem]:
    raw = [
        # ── Equations / calculations ─────────────────────────────────
        {
            "id": "eq_rsi",
            "title": "RSI equation and reading",
            "content": (
                "RSI (Relative Strength Index): RSI = 100 - (100 / (1 + RS)), "
                "where RS = Average Gain / Average Loss over N periods (usually 14). "
                "Above 70 often overbought; below 30 often oversold. On NSE, pair RSI with "
                "trend and volume — strong momentum names can stay overbought for weeks."
            ),
            "tags": ["rsi", "equation", "technical", "calculation", "indicator", "analysis"],
            "source": "bysel_builtin_v1",
        },
        {
            "id": "eq_macd",
            "title": "MACD equations",
            "content": (
                "MACD Line = EMA(12) - EMA(26); Signal = EMA(9) of MACD; Histogram = MACD - Signal. "
                "Bullish when MACD crosses above Signal; bearish when below. Avoid trading every "
                "tiny cross in sideways Nifty ranges."
            ),
            "tags": ["macd", "equation", "technical", "calculation", "ema", "analysis"],
            "source": "bysel_builtin_v1",
        },
        {
            "id": "eq_pe",
            "title": "P/E valuation equation",
            "content": (
                "P/E = Price / EPS. Trailing uses last-12-month EPS; forward uses estimates. "
                "Compare within sector (TCS vs INFY, HDFCBANK vs ICICIBANK), not across banks vs IT. "
                "Low P/E alone is not undervaluation — check growth, debt, and earnings quality."
            ),
            "tags": ["pe", "valuation", "fundamentals", "equation", "eps", "analysis"],
            "source": "bysel_builtin_v1",
        },
        {
            "id": "eq_pb",
            "title": "P/B equation",
            "content": (
                "Price-to-Book = Market Price / Book Value per Share. Useful for banks and NBFCs "
                "(HDFCBANK, SBIN, BAJFINANCE). High P/B can be justified by high ROE; low P/B may "
                "signal asset quality or profitability stress."
            ),
            "tags": ["pb", "price to book", "valuation", "fundamentals", "banking", "equation"],
            "source": "bysel_builtin_v1",
        },
        {
            "id": "eq_cagr",
            "title": "CAGR equation",
            "content": (
                "CAGR = (Ending / Beginning) ^ (1 / Years) - 1. Example: 1,00,000 to 1,61,051 in 5 years "
                "is about 10% CAGR. Use for multi-year stock or SIP returns; avoid applying CAGR to "
                "noisy 1–2 week moves."
            ),
            "tags": ["cagr", "return", "calculation", "equation", "sip"],
            "source": "bysel_builtin_v1",
        },
        {
            "id": "eq_roe_roce",
            "title": "ROE and ROCE equations",
            "content": (
                "ROE = Net Profit / Equity. ROCE = EBIT / Capital Employed (Equity + Debt - Cash approx). "
                "Prefer businesses with durable ROE/ROCE and controlled leverage. High ROE from debt alone "
                "is a red flag."
            ),
            "tags": ["roe", "roce", "fundamentals", "equation", "quality", "analysis"],
            "source": "bysel_builtin_v1",
        },
        {
            "id": "eq_eps",
            "title": "EPS equation",
            "content": (
                "EPS = Net profit attributable to equity / weighted average shares. Diluted EPS includes "
                "ESOPs and convertibles. EPS growth drives long-term equity compounding on NSE/BSE."
            ),
            "tags": ["eps", "earnings", "fundamentals", "equation"],
            "source": "bysel_builtin_v1",
        },
        {
            "id": "eq_vwap_atr",
            "title": "VWAP and ATR equations",
            "content": (
                "VWAP = sum(Price × Volume) / sum(Volume) for the session — institutional intraday benchmark. "
                "ATR averages True Range (max of high-low, |high-prev close|, |low-prev close|). Use ATR for "
                "stop distance and position sizing, not direction."
            ),
            "tags": ["vwap", "atr", "volatility", "technical", "equation", "intraday"],
            "source": "bysel_builtin_v1",
        },
        {
            "id": "eq_bollinger_sharpe",
            "title": "Bollinger Bands and Sharpe",
            "content": (
                "Bollinger: Middle = SMA(N); Upper/Lower = Middle ± k×StdDev (often N=20, k=2). "
                "Sharpe = (Return - Risk-free rate) / StdDev of returns. Higher Sharpe = better "
                "risk-adjusted performance."
            ),
            "tags": ["bollinger", "sharpe", "equation", "volatility", "risk", "calculation"],
            "source": "bysel_builtin_v1",
        },
        {
            "id": "eq_rr_drawdown",
            "title": "Risk-reward and drawdown",
            "content": (
                "Risk/Reward = (Target - Entry) / (Entry - Stop). Many traders prefer ≥ 1:2. "
                "Drawdown = (Trough - Peak) / Peak; max drawdown is the worst peak-to-trough fall. "
                "Size positions so a full stop does not ruin the portfolio."
            ),
            "tags": ["risk", "reward", "drawdown", "stop loss", "portfolio", "equation"],
            "source": "bysel_builtin_v1",
        },
        {
            "id": "eq_beta_peg",
            "title": "Beta and PEG",
            "content": (
                "Beta measures sensitivity to Nifty moves (beta > 1 amplifies market swings). "
                "PEG = P/E / expected earnings growth rate. PEG near 1 can imply fair growth-adjusted "
                "valuation, but growth estimates are uncertain."
            ),
            "tags": ["beta", "peg", "valuation", "risk", "equation", "nifty"],
            "source": "bysel_builtin_v1",
        },
        {
            "id": "eq_dividend_yield",
            "title": "Dividend yield and payout",
            "content": (
                "Dividend Yield = Annual Dividend per Share / Price. Payout Ratio = Dividends / Net Profit. "
                "PSU and mature cash businesses (COALINDIA, ITC, ONGC) often lead yield screens; growth IT "
                "usually pays less."
            ),
            "tags": ["dividend", "yield", "payout", "fundamentals", "equation", "income"],
            "source": "bysel_builtin_v1",
        },
        # ── Market mechanics / NSE-BSE-SEBI ──────────────────────────
        {
            "id": "mx_nse_bse",
            "title": "NSE and BSE basics",
            "content": (
                "NSE and BSE are India's primary equity exchanges. Most liquid stocks trade on NSE with "
                "symbols like RELIANCE, TCS, INFY. BSE uses scrip codes; both map to the same ISIN. "
                "BYSEL quotes typically use NSE symbols (for example RELIANCE, TCS)."
            ),
            "tags": ["nse", "bse", "exchange", "isin", "symbols", "stocks"],
            "source": "bysel_builtin_v1",
        },
        {
            "id": "mx_sebi",
            "title": "SEBI investor protection",
            "content": (
                "SEBI regulates securities markets, disclosures, insider trading, brokers, and mutual funds. "
                "Listed companies must publish quarterly results and material events. Treat tip lines and "
                "guaranteed-return claims as compliance red flags."
            ),
            "tags": ["sebi", "regulation", "compliance", "disclosure", "events"],
            "source": "bysel_builtin_v1",
        },
        {
            "id": "mx_circuit",
            "title": "Circuit filters and freeze",
            "content": (
                "NSE/BSE apply price bands (circuits) that halt trading beyond ±X% from the previous close "
                "or reference price. Bands vary by stock (often 5/10/20%). Index circuit breakers can pause "
                "the whole market on extreme moves. Never assume you can exit at any price during a freeze."
            ),
            "tags": ["circuit", "price band", "nse", "risk", "trading", "volatility"],
            "source": "bysel_builtin_v1",
        },
        {
            "id": "mx_session",
            "title": "Indian market session timings (CAS from Aug 2026)",
            "content": (
                "From 3 August 2026 NSE/BSE no longer share one single 3:30 PM close for every segment. "
                "Cash still opens 9:15 AM IST (pre-open ~9:00–9:15). F&O-eligible cash stocks end "
                "continuous trading at 3:15 PM and enter a Closing Auction Session (CAS) until 3:35 PM "
                "for closing-price discovery. Non-F&O cash still runs to 3:30 PM. Equity derivatives "
                "(index/stock F&O) trade until 3:40 PM. Post-close order windows follow exchange rules. "
                "Monday–Friday excluding holidays. Verify live circulars for MIS square-off times."
            ),
            "tags": [
                "market hours", "nse", "bse", "session", "trading", "ist",
                "closing auction", "cas", "timings", "f&o",
            ],
            "source": "bysel_builtin_v16",
        },
        {
            "id": "mx_settlement",
            "title": "T+1 settlement and delivery",
            "content": (
                "Indian equity cash currently settles on T+1. Delivery trades give you shares in demat; "
                "intraday/MIS square-off before close. Delivery % = delivery volume / total volume — "
                "high delivery can support genuine accumulation narratives."
            ),
            "tags": ["settlement", "delivery", "t+1", "demat", "trading"],
            "source": "bysel_builtin_v1",
        },
        {
            "id": "mx_fo",
            "title": "F&O basics India",
            "content": (
                "Futures & Options on NSE cover indices (NIFTY, BANKNIFTY) and many stocks with lot sizes. "
                "Weekly/monthly expiries, mark-to-market on futures, and option Greeks (delta, gamma, theta, vega) "
                "drive risk. F&O is leveraged — margin calls and expiry decay can wipe capital quickly."
            ),
            "tags": ["fno", "futures", "options", "nifty", "banknifty", "derivatives", "greeks"],
            "source": "bysel_builtin_v1",
        },
        {
            "id": "groww_fo_map",
            "title": "What is Futures & Options (Groww-style map)",
            "content": (
                "F&O are derivatives on an underlying (stock/index/commodity/ETF). Futures create an "
                "obligation for both sides with daily M2M and margins; options give the buyer a right "
                "(call=buy, put=sell) for a premium while the seller is obligated if exercised. "
                "Used to hedge or speculate with leverage — wrong direction can cause large losses. "
                "Index F&O is typically cash-settled; stock F&O may involve physical settlement near expiry."
            ),
            "tags": [
                "f&o", "futures and options", "derivatives", "futures vs options",
                "cash settlement", "education", "fno",
            ],
            "source": "bysel_builtin_v13",
        },
        {
            "id": "groww_fo_participants",
            "title": "F&O users: hedgers, speculators, arbitrageurs",
            "content": (
                "Hedgers lock or cushion price risk (e.g. producer sells futures; equity book shorts "
                "index futures). Speculators take direction: bullish→long futures/calls; "
                "bearish→short futures/puts — usually cash-settled intent. Arbitrageurs fade "
                "futures–spot gaps vs cost of carry toward fair value. Leverage means margin, not "
                "full notional — amplify both profit and loss."
            ),
            "tags": [
                "hedgers", "speculators", "arbitrageurs", "f&o participants",
                "cost of carry", "education", "fno",
            ],
            "source": "bysel_builtin_v13",
        },
        {
            "id": "mx_iv_greeks",
            "title": "Implied volatility and Greeks",
            "content": (
                "Implied Volatility (IV) is the market's expected move priced into options. "
                "Delta ≈ directional sensitivity; Theta = time decay; Vega = sensitivity to IV. "
                "High IV around events (RBI, results) makes options expensive — selling premium has "
                "different risk than buying lottery calls."
            ),
            "tags": ["iv", "implied volatility", "delta", "theta", "vega", "options", "equation"],
            "source": "bysel_builtin_v1",
        },
        {
            "id": "mx_fii_dii",
            "title": "FII and DII flows",
            "content": (
                "FIIs are foreign institutions; DIIs are domestic mutual funds/insurers. Persistent FII selling "
                "often pressures Nifty/banks/IT; DII SIP buying can cushion drawdowns. Flows are context, "
                "not a standalone buy/sell signal for one stock."
            ),
            "tags": ["fii", "dii", "flows", "nifty", "institutional", "macro"],
            "source": "bysel_builtin_v1",
        },
        {
            "id": "mx_indices",
            "title": "Key Indian indices and symbols",
            "content": (
                "NIFTY50 tracks 50 large NSE names; SENSEX is BSE's 30-stock index; BANKNIFTY tracks banks. "
                "Sector indices (IT, Pharma, Auto, FMCG, Energy, Realty, PSU Bank) help relative strength. "
                "In BYSEL, index symbols include NIFTY50, SENSEX, BANKNIFTY."
            ),
            "tags": ["nifty", "sensex", "banknifty", "indices", "symbols", "stocks"],
            "source": "bysel_builtin_v1",
        },
        # ── Analysis frameworks ──────────────────────────────────────
        {
            "id": "an_checklist",
            "title": "Stock analysis checklist",
            "content": (
                "Practical analysis stack: (1) Business quality — ROE/ROCE, margins, cash flow. "
                "(2) Valuation — P/E, P/B, PEG vs peers/history. (3) Technical — trend, RSI/MACD, support/resistance. "
                "(4) Events — results, guidance, promoter pledge, regulatory news. (5) Risk — debt, circuit, liquidity. "
                "Score conviction only after at least three pillars agree."
            ),
            "tags": ["analysis", "fundamental", "technical", "checklist", "stocks"],
            "source": "bysel_builtin_v1",
        },
        {
            "id": "an_support_resistance",
            "title": "Support and resistance",
            "content": (
                "Support is a price zone where buying historically appears; resistance where selling appears. "
                "Use prior swing highs/lows, round numbers, and volume nodes. Breakouts need volume confirmation; "
                "failed breakouts often reverse quickly on NSE momentum names."
            ),
            "tags": ["support", "resistance", "technical", "analysis", "price action"],
            "source": "bysel_builtin_v1",
        },
        {
            "id": "an_prediction_guard",
            "title": "Prediction guardrails",
            "content": (
                "Price forecasts must be probabilistic with ranges, not guarantees. Combine trend, valuation, "
                "and catalysts; state invalidation (stop) clearly. Never promise returns. Prefer scenario "
                "analysis: bull / base / bear with catalysts for each."
            ),
            "tags": ["prediction", "forecast", "risk", "uncertainty", "guidance"],
            "source": "bysel_builtin_v1",
        },
        {
            "id": "an_portfolio",
            "title": "Portfolio construction India",
            "content": (
                "Diversify across sectors (banking, IT, consumer, energy, pharma) and market caps. "
                "Avoid single-stock concentration above a risk budget. Rebalance periodically; use SIP for "
                "core equity and satellites for high-conviction ideas with stops."
            ),
            "tags": ["portfolio", "diversification", "risk", "sip", "allocation"],
            "source": "bysel_builtin_v1",
        },
        # ── Sector primers ───────────────────────────────────────────
        {
            "id": "sec_banking",
            "title": "Banking sector primer",
            "content": (
                "Private banks (HDFCBANK, ICICIBANK, AXISBANK, KOTAKBANK) and PSU banks (SBIN) are sensitive to "
                "RBI rates, credit growth, NIM, GNPA/NNPA, and CASA. Rate cuts can help margins with lag; "
                "asset-quality shocks re-rate P/B quickly."
            ),
            "tags": ["banking", "rbi", "hdfcbank", "sbin", "nim", "fundamentals", "sector"],
            "source": "bysel_builtin_v1",
        },
        {
            "id": "sec_it",
            "title": "IT sector primer",
            "content": (
                "IT majors (TCS, INFY, WIPRO, HCLTECH, TECHM) depend on US/EU demand, deal wins, attrition, "
                "and currency. A weak rupee can aid reported revenue; discretionary tech spend slowdowns "
                "hurt midcaps first. Watch guidance and large-deal pipelines."
            ),
            "tags": ["it", "tcs", "infy", "wipro", "earnings", "sector", "guidance"],
            "source": "bysel_builtin_v1",
        },
        {
            "id": "sec_pharma",
            "title": "Pharma sector primer",
            "content": (
                "Pharma names (SUNPHARMA, DRREDDY, CIPLA, DIVISLAB, LUPIN) mix domestic formulations, US generics, "
                "and API/CDMO. FDA observations, price erosion, and product approvals drive swings. "
                "Export mix and currency matter for margins."
            ),
            "tags": ["pharma", "sunpharma", "cipla", "sector", "earnings"],
            "source": "bysel_builtin_v1",
        },
        {
            "id": "sec_auto_energy",
            "title": "Auto and energy primers",
            "content": (
                "Auto (TMPV, TMCV, MARUTI, M&M) tracks volumes, rural demand, EV transition, and commodity costs. "
                "Note: TATAMOTORS is no longer listed — PV/JLR trades as TMPV; commercial vehicles as TMCV. "
                "Energy/PSU (RELIANCE, ONGC, NTPC, POWERGRID, COALINDIA) mixes refining/marketing, upstream, "
                "and regulated utilities — sensitive to crude, tarifs, and government policy."
            ),
            "tags": ["auto", "energy", "reliance", "maruti", "ntpc", "sector"],
            "source": "bysel_builtin_v1",
        },
        {
            "id": "sec_fmcg_defence",
            "title": "FMCG and defence primers",
            "content": (
                "FMCG (HINDUNILVR, ITC, NESTLEIND, BRITANNIA) focuses on volume growth, rural recovery, and margins. "
                "Defence (HAL, BEL, BDL) is order-book and execution driven with government capex cycles — "
                "valuations can stay elevated on multi-year visibility."
            ),
            "tags": ["fmcg", "defence", "hal", "bel", "itc", "sector"],
            "source": "bysel_builtin_v1",
        },
        {
            "id": "sec_defence_names_v2",
            "title": "Top defence / aerospace liquid names (India)",
            "content": (
                "Commonly researched NSE defence/aerospace names: HAL, BEL, BDL, MAZDOCK, COCHINSHIP, GRSE, "
                "DATAPATTNS, SOLARINDS (propellants/explosives adjacency), Bharat Dynamics (BDL) for missiles. "
                "Drivers: MoD order book, indigenisation (Atmanirbhar), export clearances, and quarterly execution. "
                "This is a research universe — not a ranked buy list. Prefer names with liquidity and disclosed orders."
            ),
            "tags": [
                "defence", "defense", "hal", "bel", "bdl", "mazdock", "sector", "top", "stocks", "screen",
            ],
            "source": "bysel_builtin_v2",
        },
        {
            "id": "cmp_it_peers_v2",
            "title": "Comparing TCS vs INFY vs WIPRO",
            "content": (
                "IT peer compare checklist: (1) revenue growth & large-deal TCV, (2) EBIT margin trend, "
                "(3) attrition/utilization, (4) geographic mix (US discretionary risk), (5) valuation (P/E, PEG) "
                "vs growth, (6) rupee sensitivity. TCS often priced for quality/stability; INFY for growth narrative; "
                "WIPRO more turnaround/valuation sensitive. Prefer the cleaner earnings + guidance combo for your horizon."
            ),
            "tags": ["compare", "tcs", "infy", "wipro", "it", "valuation", "peers"],
            "source": "bysel_builtin_v2",
        },
        {
            "id": "cmp_scope_named_only_v1",
            "title": "Compare only the stocks the user named",
            "content": (
                "When the user asks to compare two or three named tickers (e.g. TMPV with MARUTI, TCS vs INFY), "
                "the scorecard must include only those names. Do not inject portfolio holdings, selected-quote "
                "context, or unrelated large-caps (HCLTECH, ICICIBANK, etc.). "
                "Stop-loss / entry / target questions for a named stock (e.g. stop loss for INFY swing) need that "
                "stock's paper levels — not a generic glossary definition of stop-loss."
            ),
            "tags": [
                "compare", "tmpv", "maruti", "stop loss", "entry", "target",
                "context", "holdings", "scorecard", "education",
            ],
            "source": "bysel_builtin_v14",
        },
        # ── Symbol / listing literacy ────────────────────────────────
        {
            "id": "sym_literacy",
            "title": "Reading Indian stock symbols",
            "content": (
                "NSE equity symbols are usually uppercase tickers (RELIANCE, HDFCBANK). Prefer exact symbols in "
                "queries for BYSEL. NSE names are the usual tickers; BSE uses scrip codes. "
                "Indices use special tickers (NIFTY50, SENSEX, BANKNIFTY). Always verify company name when "
                "symbols look similar (e.g. HDFCBANK vs HDFCLIFE)."
            ),
            "tags": ["symbols", "ticker", "nse", "bse", "stocks", "search"],
            "source": "bysel_builtin_v1",
        },
        {
            "id": "sym_bluechips",
            "title": "Common liquid large-cap symbols",
            "content": (
                "Highly traded large caps include RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK, SBIN, ITC, LT, "
                "BHARTIARTL, AXISBANK, KOTAKBANK, BAJFINANCE, HINDUNILVR, SUNPHARMA, MARUTI, TITAN, NTPC, "
                "POWERGRID, ULTRACEMCO, TMPV, TATASTEEL, WIPRO, HCLTECH. These dominate Nifty weight "
                "and BYSEL default watchlists."
            ),
            "tags": ["symbols", "nifty", "largecap", "stocks", "watchlist"],
            "source": "bysel_builtin_v1",
        },
        {
            "id": "term_sip_ipo",
            "title": "SIP and IPO terms",
            "content": (
                "SIP invests a fixed amount regularly into mutual funds for rupee-cost averaging. "
                "IPO lists a private company on NSE/BSE under SEBI rules; retail often uses UPI/ASBA and "
                "may face lottery allotment when oversubscribed. Listing gains are never guaranteed."
            ),
            "tags": ["sip", "ipo", "mutual fund", "sebi", "terms"],
            "source": "bysel_builtin_v1",
        },
        {
            "id": "term_order_types",
            "title": "Order types and charges context",
            "content": (
                "Market orders fill at available price; limit orders set your price. SL/SL-M help risk control. "
                "Indian trades incur brokerage, STT, exchange fees, GST, and stamp duty — always check "
                "pre-trade estimates before sizing."
            ),
            "tags": ["orders", "stop loss", "charges", "stt", "trading", "terms"],
            "source": "bysel_builtin_v1",
        },
        {
            "id": "term_promoter_pledge",
            "title": "Promoter holding and pledging",
            "content": (
                "Promoter holding shows skin in the game. High promoter pledging can be a stress signal if "
                "prices fall and lenders force sales. Track shareholding pattern changes and pledged "
                "percentage in quarterly disclosures."
            ),
            "tags": ["promoter", "pledge", "shareholding", "risk", "fundamentals"],
            "source": "bysel_builtin_v1",
        },
        # ── bysel_builtin_v2 — Hinglish retail jargon ─────────────────
        {
            "id": "v2_hinglish_demat",
            "title": "Demat account (Hinglish retail jargon)",
            "content": (
                "Demat = dematerialised account jahan shares electronic form mein rehte hain (NSDL/CDSL). "
                "Delivery kharidne ke baad shares demat mein credit hote hain (T+1 settlement). "
                "Trading account orders place karta hai; demat holdings store karta hai. "
                "BYSEL paper trades simulate karti hain — real demat debit/credit nahi hota."
            ),
            "tags": ["demat", "delivery", "settlement", "hinglish", "terms", "trading"],
            "source": "bysel_builtin_v2",
        },
        {
            "id": "v2_hinglish_circuit",
            "title": "Circuit / price band (Hinglish)",
            "content": (
                "Circuit ya price band matlab stock ±X% se zyada move nahi kar sakta us din "
                "(often 5/10/20% depending on stock). Upper circuit pe buyers, lower pe sellers freeze. "
                "Freeze ke dauran exit guarantee nahi — liquidity gayab ho sakti hai. "
                "Index circuit breakers pure market ko pause kar sakte hain extreme moves par."
            ),
            "tags": ["circuit", "price band", "hinglish", "risk", "nse", "volatility"],
            "source": "bysel_builtin_v2",
        },
        {
            "id": "v2_hinglish_t1_stt",
            "title": "T+1, STT, delivery vs intraday (Hinglish)",
            "content": (
                "T+1: aaj delivery kharido, next trading day demat settle. "
                "STT (Securities Transaction Tax) equity cash/FO trades par lagta hai — brokerage alag hai. "
                "Delivery = shares rakhna (overnight risk + full STT on sell side typically higher framing). "
                "Intraday/MIS = same-day square-off; overnight carry nahi. "
                "Educational only — exact charge schedule broker/exchange ke hisaab se verify karo."
            ),
            "tags": ["t+1", "stt", "delivery", "intraday", "hinglish", "charges", "settlement"],
            "source": "bysel_builtin_v2",
        },
        # ── F&O education ────────────────────────────────────────────
        {
            "id": "v2_fo_lot_expiry",
            "title": "F&O lot size and expiry",
            "content": (
                "NSE F&O contracts trade in fixed lot sizes (e.g. NIFTY lots change over time; stock lots vary). "
                "One order = one or more lots, not single shares. Expiry is weekly or monthly depending on "
                "the contract — at expiry futures settle to cash/spot rules and options may expire worthless. "
                "Rolling near expiry avoids gamma/theta spikes but adds cost. Always check current lot size "
                "on NSE before sizing risk."
            ),
            "tags": ["fno", "lot", "expiry", "futures", "options", "nifty", "derivatives"],
            "source": "bysel_builtin_v2",
        },
        {
            "id": "v2_fo_margin_risk",
            "title": "F&O margin and leverage risk",
            "content": (
                "Margin lets you control a large notional with less cash — SPAN + exposure margins apply. "
                "MTM losses on futures can trigger margin calls the same day. Option buyers risk premium; "
                "sellers face potentially large losses. Never size F&O like cash equity: a few lots can "
                "wipe a small account on a gap. Prefer paper practice of lot risk before any live F&O."
            ),
            "tags": ["fno", "margin", "leverage", "risk", "futures", "options", "derivatives"],
            "source": "bysel_builtin_v2",
        },
        # ── Equity tax basics (educational) ──────────────────────────
        {
            "id": "v2_tax_stcg_ltcg",
            "title": "Equity STCG and LTCG basics (educational)",
            "content": (
                "For listed equity (and equity-oriented funds under common retail framing): "
                "holding ≤ 12 months is often treated as short-term (STCG); > 12 months as long-term (LTCG). "
                "Rates and exemptions change with Budget/Finance Act — e.g. historical LTCG exemption "
                "thresholds and surcharge rules. STT-paid delivery equity has a different tax path than "
                "intraday/F&O (business income). This is high-level education only — confirm current slabs "
                "with a CA/tax professional; BYSEL does not compute your tax liability."
            ),
            "tags": ["stcg", "ltcg", "tax", "equity", "delivery", "education", "sebi"],
            "source": "bysel_builtin_v2",
        },
        # ── Corporate actions literacy ───────────────────────────────
        {
            "id": "v2_corporate_actions",
            "title": "Corporate actions literacy",
            "content": (
                "Common actions: dividend (cash credit), bonus (free shares, price adjusts), "
                "stock split (more shares, lower price, same value), rights issue (offer to buy more), "
                "buyback, merger/demerger. Record date / ex-date decide who gets the benefit. "
                "Charts often adjust for splits/bonus — compare adjusted vs unadjusted carefully. "
                "Always read exchange/company notices; do not trade only on tip-style rumour of actions."
            ),
            "tags": ["corporate actions", "dividend", "bonus", "split", "rights", "events", "education"],
            "source": "bysel_builtin_v2",
        },
        # ── Paper-practice coaching ──────────────────────────────────
        {
            "id": "v2_paper_journal",
            "title": "Paper-trade journaling (entry / SL / target)",
            "content": (
                "BYSEL defaults to paper/simulation money — practice process, not tip-chasing. "
                "Before each paper trade write: (1) why this setup, (2) entry zone, (3) stop-loss, "
                "(4) target / R:R, (5) invalidation. After exit: what worked, what you ignored, "
                "did you move SL emotionally? Never chase a missed move — wait for the next valid setup. "
                "Journaling builds discipline; random 'sure tip' entries do not."
            ),
            "tags": ["paper", "practice", "journal", "stop loss", "entry", "target", "education", "risk"],
            "source": "bysel_builtin_v2",
        },
        {
            "id": "v2_paper_no_chase",
            "title": "Practice stance: no chase, levels first",
            "content": (
                "Practice coaching: mark support/resistance or VWAP levels first, then decide. "
                "If price already ran far from your planned entry, skip — FOMO is not a plan. "
                "Size paper positions so a full stop is emotionally boring. Review weekly win-rate "
                "and average R — process metrics beat one lucky paper P&L screenshot."
            ),
            "tags": ["paper", "practice", "levels", "support", "resistance", "discipline", "education"],
            "source": "bysel_builtin_v2",
        },
        # ── Portfolio practice stance ────────────────────────────────
        {
            "id": "v2_portfolio_practice",
            "title": "Portfolio practice stance (educational)",
            "content": (
                "In paper portfolios: diversify across sectors, cap single-name risk, and separate "
                "core (SIP-like slow adds) from satellite swing ideas with hard stops. "
                "Rebalance on rules (e.g. sector drift), not headlines. Track drawdown of the whole "
                "book, not only winners. Practice stance = risk budget first, conviction second — "
                "educational simulation, not a live advisory mandate."
            ),
            "tags": ["portfolio", "practice", "diversification", "sip", "risk", "allocation", "education"],
            "source": "bysel_builtin_v2",
        },
        # ── Beginner market literacy (how the market works) ───────────
        {
            "id": "mx_stock_market_meaning",
            "title": "Stock market meaning (India)",
            "content": (
                "The stock market is a regulated platform where publicly listed companies' shares "
                "(stocks/equities) are bought and sold. Companies issue shares to raise capital; buyers "
                "become shareholders with residual ownership claims. In India, cash equities mainly trade "
                "on NSE and BSE under SEBI rules. Education only — not a tip to buy any stock."
            ),
            "tags": [
                "stock market", "meaning", "shares", "equity", "nse", "bse", "beginner", "education",
                "how it works",
            ],
            "source": "bysel_builtin_v3",
        },
        {
            "id": "mx_how_market_works_india",
            "title": "How the Indian stock market works (5 steps)",
            "content": (
                "Beginner flow: (1) Primary market — company raises money via IPO (SEBI-regulated). "
                "(2) Listing — shares list on NSE and/or BSE for secondary trading. "
                "(3) Broker/app — investors place buy/sell via a stockbroker (trading + demat). "
                "(4) Order matching — exchange matching engine pairs compatible bids and offers in real time. "
                "(5) Settlement — equity cash settles on T+1; shares credit/debit demat, funds settle to "
                "trading/bank ledger. BYSEL paper trades simulate process; they do not move real demat."
            ),
            "tags": [
                "how does the stock market work", "how it works", "india", "ipo", "listing", "broker",
                "order matching", "settlement", "t+1", "beginner", "education", "nse", "bse",
            ],
            "source": "bysel_builtin_v3",
        },
        {
            "id": "mx_participants",
            "title": "Key stock market participants (India)",
            "content": (
                "Participants: (1) Retail investors — personal capital, often longer horizon. "
                "(2) Traders — shorter-horizon price moves (intraday/swing). "
                "(3) Institutional investors — mutual funds, insurers, FIIs/FPIs, with large tickets. "
                "(4) Exchanges — NSE/BSE match orders and publish prices. "
                "(5) Depositories — NSDL and CDSL hold securities in demat form. "
                "(6) Depository Participants (DPs) — brokers/banks that open/maintain demat and link to "
                "depositories. (7) SEBI — regulator for investor protection, brokers, disclosures, fraud."
            ),
            "tags": [
                "participants", "retail", "trader", "institutional", "fii", "dii", "nse", "bse",
                "nsdl", "cdsl", "depository", "dp", "broker", "sebi", "beginner", "education",
            ],
            "source": "bysel_builtin_v3",
        },
        {
            "id": "mx_nsdl_cdsl_dp",
            "title": "NSDL, CDSL and Depository Participants",
            "content": (
                "NSDL and CDSL are India's securities depositories — they keep shares electronic. "
                "You do not open demat directly at NSDL/CDSL for retail; a Depository Participant (DP) — "
                "usually your broker or bank — opens and maintains the demat account and instructs the "
                "depository on transfers. Trading account places orders; demat holds settled delivery shares. "
                "Confirm DP charges (AMC) with your broker."
            ),
            "tags": [
                "nsdl", "cdsl", "depository", "dp", "depository participant", "demat", "broker",
                "beginner", "education",
            ],
            "source": "bysel_builtin_v3",
        },
        {
            "id": "mx_price_discovery",
            "title": "How share prices are determined",
            "content": (
                "Primary driver is demand vs supply on the exchange order book: more aggressive buying "
                "lifts price; more selling pressures it. Influencers: company performance (revenue, margins, "
                "EPS, guidance), macro (rates, inflation, GDP, policy), sector trends, news/events, "
                "investor sentiment (fear/greed), institutional flows (FII/DII/MF), and liquidity "
                "(thin names gap more). Short-term moves can be emotional; long-term prices still orbit "
                "cash flows and growth. Educational framing — not a prediction model."
            ),
            "tags": [
                "share price", "price discovery", "demand", "supply", "liquidity", "sentiment",
                "fundamentals", "macro", "news", "beginner", "education", "how prices",
            ],
            "source": "bysel_builtin_v3",
        },
        {
            "id": "mx_start_investing_india",
            "title": "How to start investing in the Indian share market",
            "content": (
                "Educational checklist (not onboarding advice for any broker): "
                "(1) Open trading + demat via a registered broker/DP. "
                "(2) Complete KYC — typically PAN, Aadhaar, bank proof, photo/signature (confirm current rules). "
                "(3) Link bank and add funds to the trading ledger. "
                "(4) Research business quality, financials, and industry — not tips. "
                "(5) Place a limit/market order for quantity and price you understand. "
                "(6) Start small; size so a loss is survivable. "
                "(7) Diversify across sectors/names. "
                "(8) Review periodically vs your goal/risk. "
                "BYSEL is best used for paper practice of steps 4–8 before live capital."
            ),
            "tags": [
                "start investing", "how to start", "kyc", "demat", "trading account", "beginner",
                "diversify", "education", "india", "share market",
            ],
            "source": "bysel_builtin_v3",
        },
        {
            "id": "mx_common_mistakes",
            "title": "Common share-market mistakes to avoid",
            "content": (
                "Common beginner mistakes: investing without basic knowledge; following tips/rumours; "
                "overtrading; ignoring diversification; trying to time every tick perfectly; letting "
                "fear or greed override a written plan; having no clear goal or risk budget; "
                "confusing paper wins with live edge; chasing after big moves. "
                "Process fix: written thesis, entry, stop, invalidation — then review. Educational only."
            ),
            "tags": [
                "mistakes", "common mistakes", "tips", "rumour", "overtrading", "diversification",
                "fear", "greed", "discipline", "beginner", "education", "risk",
            ],
            "source": "bysel_builtin_v3",
        },
        {
            "id": "mx_primary_secondary",
            "title": "Primary vs secondary market",
            "content": (
                "Primary market: company issues new shares (IPO/FPO/rights) and receives capital. "
                "Secondary market: investors trade existing listed shares with each other on NSE/BSE; "
                "the company typically does not receive that trade cash. Most daily price charts you see "
                "are secondary-market trading."
            ),
            "tags": [
                "primary market", "secondary market", "ipo", "listing", "nse", "bse", "beginner",
                "education",
            ],
            "source": "bysel_builtin_v3",
        },
        # ── Zerodha Varsity Ch.6-aligned literacy ────────────────────
        {
            "id": "mx_public_limited",
            "title": "Public limited company & disclosure",
            "content": (
                "After an IPO, a public limited company must disclose material information to the public "
                "(quarterly results, material events under SEBI LODR). Its shares trade daily on exchanges. "
                "Listed status means continuous disclosure obligations — not a guarantee of returns. "
                "Educational framing inspired by standard India market literacy (e.g. Varsity Module 1)."
            ),
            "tags": [
                "public limited", "listed company", "disclosure", "ipo", "sebi", "beginner", "education",
            ],
            "source": "bysel_builtin_v4",
        },
        {
            "id": "mx_different_opinions",
            "title": "Different opinions make a market",
            "content": (
                "A stock market is an electronic marketplace where buyers and sellers express opposing "
                "views as orders. Same news can make A a seller and B a buyer — that disagreement is "
                "what creates a trade. Brokers route orders; the exchange matching engine pairs them. "
                "No opposing view → no trade at your price. Key takeaway: markets need two sides."
            ),
            "tags": [
                "stock market", "opinions", "buyer", "seller", "order matching", "beginner", "education",
                "what is the stock market",
            ],
            "source": "bysel_builtin_v4",
        },
        {
            "id": "mx_what_moves_stock",
            "title": "What moves stock prices (news, sector, liquidity)",
            "content": (
                "Prices move when participants react to news/events — company-specific (CEO, results), "
                "industry (sector association commentary), or economy/politics. In a bullish tape, buyers "
                "often lift offers and pay up; prices can jump in minutes. Illiquid unknown names may "
                "barely move with no news; large liquid names (e.g. RELIANCE) still trade on demand/supply "
                "even on quiet news days. Expectation of news also moves prices. Educational — not a "
                "trading signal."
            ),
            "tags": [
                "what moves stock", "news", "events", "bullish", "liquidity", "sector", "demand",
                "supply", "beginner", "education", "price",
            ],
            "source": "bysel_builtin_v4",
        },
        {
            "id": "mx_trade_lifecycle",
            "title": "How a stock trade is executed (buy path)",
            "content": (
                "To buy: log into trading account → place order (symbol, price, quantity). Broker checks "
                "sufficient funds, then routes to the exchange. Matching can fill from one seller or many "
                "(e.g. 200 shares from ten sellers of 20). After execution, shares credit your demat and "
                "debit the seller's demat on settlement (equity cash T+1). BYSEL paper trades simulate "
                "this process without real demat movement."
            ),
            "tags": [
                "how stock gets traded", "trade execution", "broker", "order", "demat", "matching",
                "beginner", "education", "t+1",
            ],
            "source": "bysel_builtin_v4",
        },
        {
            "id": "mx_owning_stock_privileges",
            "title": "What happens after you own a stock",
            "content": (
                "After delivery settlement, shares sit in demat — you are a fractional owner of the company "
                "proportional to shares held. Ownership can entitle you to corporate benefits: dividends, "
                "bonuses, stock splits, rights issues, buybacks (when announced), and voting rights where "
                "applicable. Small retail holdings are tiny % of large caps — still real ownership, not a tip."
            ),
            "tags": [
                "own stock", "shareholder", "dividend", "bonus", "split", "rights", "voting",
                "corporate actions", "demat", "beginner", "education",
            ],
            "source": "bysel_builtin_v4",
        },
        {
            "id": "mx_holding_period",
            "title": "Holding period: minutes to forever",
            "content": (
                "Holding period is how long you intend to keep a position — from minutes (scalp/intraday) "
                "to days/weeks (swing) to years (investing). There is no universal 'correct' period; it must "
                "match your process and risk. Legendary long-term investors often prefer multi-year holds; "
                "traders intentionally keep holds short. Pick a style before sizing risk."
            ),
            "tags": [
                "holding period", "investor", "trader", "swing", "intraday", "long term", "beginner",
                "education",
            ],
            "source": "bysel_builtin_v4",
        },
        {
            "id": "mx_absolute_vs_cagr",
            "title": "Absolute return vs CAGR (when to use which)",
            "content": (
                "Absolute return = (End/Start − 1)×100 — use for holds of about one year or less. "
                "Example: buy 3030, sell 3550 → ~17.16% absolute. "
                "CAGR = (End/Start)^(1/years) − 1 — use to compare multi-year growth rates apples-to-apples. "
                "Same 3030→3550 over 2 years ≈ 8.2% CAGR. For sub-year wins, annualize carefully "
                "(e.g. 17% in 6 months is not automatically a safe '34% year' expectation). "
                "Ask BYSEL: 'CAGR of 3030 to 3550 in 2 years' or 'return from 3030 to 3550'."
            ),
            "tags": [
                "absolute return", "cagr", "returns", "annualized", "calculation", "equation",
                "beginner", "education", "how to calculate returns",
            ],
            "source": "bysel_builtin_v4",
        },
        {
            "id": "mx_trader_investor_styles",
            "title": "Trader vs investor styles (where do you fit)",
            "content": (
                "Traders seek shorter opportunities and manage risk actively: "
                "Day trader — open and close same day (no overnight); "
                "Scalper — many quick trades, small ticks, large size; "
                "Swing trader — holds days to weeks. "
                "Investors accept longer evolution: "
                "Growth investor — emerging industry/macro growth stories; "
                "Value investor — quality businesses temporarily beaten down. "
                "Style = holding period + risk tolerance. BYSEL paper practice helps discover fit — "
                "not a recommendation of any style."
            ),
            "tags": [
                "trader", "investor", "day trader", "scalper", "swing trader", "growth", "value",
                "style", "where do you fit", "beginner", "education", "risk",
            ],
            "source": "bysel_builtin_v4",
        },
        # ── Technical Analysis literacy (Varsity TA module-aligned) ──
        {
            "id": "ta_vs_fa",
            "title": "Technical vs fundamental analysis",
            "content": (
                "Fundamental analysis studies business quality, earnings, and valuation to estimate "
                "what a stock may be worth. Technical analysis studies price/volume history — charts, "
                "candles, indicators — to frame timing and risk. They answer different questions; many "
                "use FA for what to watch and TA for when/how to size risk. TA trading expects frequent "
                "small edges with stops — not guaranteed multi-bagger returns. Educational only."
            ),
            "tags": [
                "technical analysis", "fundamental analysis", "ta", "fa", "charts", "beginner",
                "education", "background",
            ],
            "source": "bysel_builtin_v5",
        },
        {
            "id": "ta_charts_candles",
            "title": "Chart types and candlesticks",
            "content": (
                "OHLC summarizes each bar: Open, High, Low, Close. Line charts show closes only; "
                "bar charts show OHLC; candlesticks show the same with a filled/hollow body between "
                "open and close plus wicks to high/low — easier to read sentiment at a glance. "
                "Traders often prefer candles because body vs wick quickly shows conviction vs indecision. "
                "Patterns are probabilistic, not guarantees."
            ),
            "tags": [
                "candlestick", "chart types", "ohlc", "bar chart", "line chart", "technical",
                "education", "beginner",
            ],
            "source": "bysel_builtin_v5",
        },
        {
            "id": "ta_single_candles",
            "title": "Single candlestick patterns (Marubozu, Doji, Hammer)",
            "content": (
                "Bullish Marubozu: long green body, tiny/no wicks — strong buying. "
                "Bearish Marubozu: long red body — strong selling. "
                "Doji: open≈close — indecision; context matters (after a rally can warn of pause). "
                "Spinning top: small body, longer wicks — tug of war. "
                "Hammer: small body near high, long lower wick after decline — potential bullish reversal cue. "
                "Hanging man: similar shape after a rally — potential bearish warning. "
                "Always confirm with trend/volume; paper-practice setups with stop beyond the wick."
            ),
            "tags": [
                "marubozu", "doji", "spinning top", "hammer", "hanging man", "candlestick",
                "pattern", "technical", "education",
            ],
            "source": "bysel_builtin_v5",
        },
        {
            "id": "ta_multi_candles",
            "title": "Multiple candlestick patterns (Engulfing, Harami, Stars)",
            "content": (
                "Bullish engulfing: green candle body fully covers prior red body — buyers seize control. "
                "Bearish engulfing: opposite. "
                "Bullish harami: small green inside prior large red — selling may be exhausting. "
                "Bearish harami: small red inside prior large green. "
                "Morning star: down move, small indecision, then strong up — potential bullish reversal. "
                "Evening star: opposite at tops. "
                "Gaps: open beyond prior high/low; common around news — treat as context, not magic. "
                "Educational pattern literacy — not auto buy/sell tips."
            ),
            "tags": [
                "engulfing", "harami", "morning star", "evening star", "gap", "candlestick",
                "pattern", "technical", "education",
            ],
            "source": "bysel_builtin_v5",
        },
        {
            "id": "ta_volume_price",
            "title": "Volume with price (technical)",
            "content": (
                "Volume validates price moves: rising price + rising volume often supports trend "
                "continuation; rising price + falling volume can warn of weak rally. "
                "Breakouts through resistance prefer expanding volume; low-volume breakouts fail more often. "
                "In India, also glance at delivery % when available — high delivery can support accumulation "
                "narratives vs pure intraday churn. BYSEL computes volume z-score on live series."
            ),
            "tags": [
                "volume", "price", "delivery", "breakout", "technical", "education", "confirmation",
            ],
            "source": "bysel_builtin_v5",
        },
        {
            "id": "ta_moving_average_system",
            "title": "Moving average trend system (educational)",
            "content": (
                "Moving averages smooth noise to show trend. Simple idea: price above rising SMA/EMA "
                "favors longs; below favors caution/shorts for traders. Common pairs: 20/50 for swing, "
                "50/200 for longer trend (golden/death cross heuristics). Lag is the drawback — MAs are "
                "late in chop. Combine with S/R and volume; BYSEL surfaces live MA context in enrich."
            ),
            "tags": [
                "moving average", "sma", "ema", "golden cross", "trend", "technical", "education",
            ],
            "source": "bysel_builtin_v5",
        },
        {
            "id": "ta_fibonacci",
            "title": "Fibonacci retracements (technical)",
            "content": (
                "Fibonacci sequence yields ratios used on charts: common retracements 23.6%, 38.2%, "
                "50%, 61.8% (golden-ratio cousin), and extensions like 161.8%. Traders map a swing high "
                "to swing low and watch reactions at these zones for pullback entries — confluence with "
                "S/R or MA improves odds. BYSEL computes Fib levels in the quant pack when OHLCV loads. "
                "Not magic numbers; educational confluence tool only."
            ),
            "tags": [
                "fibonacci", "retracement", "golden ratio", "technical", "education", "swing",
            ],
            "source": "bysel_builtin_v5",
        },
        {
            "id": "ta_dow_theory",
            "title": "Dow Theory trends (educational)",
            "content": (
                "Dow Theory frames market movement in trends: Primary (months–years, main tide), "
                "Secondary (weeks–months, corrective waves against the primary), and Minor (days, noise). "
                "Trends have phases; confirmation across related averages was classic Dow thinking. "
                "Trading ranges and flag-like consolidations often pause trends before continuation or "
                "break. Pair trend read with risk:reward — if R:R is poor, skip. Educational framework."
            ),
            "tags": [
                "dow theory", "primary trend", "secondary trend", "minor trend", "flag", "range",
                "technical", "education", "trend",
            ],
            "source": "bysel_builtin_v5",
        },
        {
            "id": "ta_cpr",
            "title": "Central Pivot Range (CPR)",
            "content": (
                "CPR uses prior session High/Low/Close: Pivot P=(H+L+C)/3, BC=(H+L)/2, TC=2P−BC. "
                "Price above TC is often read as bullish bias for the session; below BC bearish; "
                "inside the range = balance. Narrow CPR width is watched by some intraday traders as "
                "a potential expansion day. BYSEL computes CPR in the quant pack alongside classic pivots. "
                "Educational levels — not a standalone system."
            ),
            "tags": [
                "cpr", "central pivot range", "pivot", "tc", "bc", "intraday", "technical", "education",
            ],
            "source": "bysel_builtin_v5",
        },
        {
            "id": "ta_daily_checklist",
            "title": "Daily technical analysis checklist (get started)",
            "content": (
                "Paper-practice daily TA loop: (1) Mark higher-timeframe trend (Dow-style primary/secondary). "
                "(2) Note key S/R, VWAP/MA, CPR/pivots. (3) Scan candles for clear patterns near levels — "
                "ignore clutter mid-range. (4) Check volume/delivery confirmation. (5) RSI/MACD only as "
                "secondary timing. (6) Write entry, stop, target, R:R before clicking. (7) If R:R < ~1.5 "
                "or news risk unclear, skip. Journal after. Educational process — not tips."
            ),
            "tags": [
                "technical analysis", "checklist", "daily", "get started", "paper", "practice",
                "education", "risk reward",
            ],
            "source": "bysel_builtin_v5",
        },
        {
            "id": "mx_sentiment_analysis",
            "title": "Sentiment analysis for Indian stocks (BYSEL stack)",
            "content": (
                "BYSEL sentiment analysis is a multi-factor educational score for a stock or the market: "
                "news headline tone, RSI/momentum, Supertrend/MA trend, MACD histogram, volume confirmation, "
                "relative strength vs Nifty, and short-term ROC. Labels range from Bullish to Bearish with a "
                "composite score from -1 to +1. Extreme bullish readings can mean crowded optimism — always "
                "confirm with structure, levels, and risk. Not a prediction or SEBI RA tip. Ask "
                "'sentiment of RELIANCE' or 'market sentiment today' for a live card; stock analysis answers "
                "include a Sentiment analysis section automatically."
            ),
            "tags": [
                "sentiment", "sentiment analysis", "market sentiment", "news sentiment",
                "bullish", "bearish", "mood", "headlines", "education",
            ],
            "source": "bysel_builtin_v15",
        },
        {
            "id": "mx_tata_motors_demerger",
            "title": "TATAMOTORS delisted — use TMPV / TMCV",
            "content": (
                "TATAMOTORS is no longer a listed NSE equity symbol after the Tata Motors demerger. "
                "Passenger vehicles (incl. EV/JLR sleeve) trade as TMPV (Tata Motors Passenger Vehicles). "
                "Commercial vehicles trade as TMCV. If a user says TATAMOTORS or 'Tata Motors', "
                "default analysis to TMPV and mention TMCV for the CV business."
            ),
            "tags": [
                "tmpv", "tmcv", "tatamotors", "tata motors", "demerger",
                "auto", "symbols", "education",
            ],
            "source": "bysel_builtin_v14",
        },
        {
            "id": "mx_renamed_tickers",
            "title": "Common renamed NSE tickers (use current symbols)",
            "content": (
                "Prefer current listings: ADANITRANS→ADANIENSOL, GMRINFRA→GMRAIRPORT, "
                "MINDAIND→UNOMINDA, MOTHERSUMI→MOTHERSON, L&TFH→LTF, IIFLWAM→360ONE, "
                "IBULHSGFIN→SAMMAANCAP, ADANIWILMAR→AWL, HPCL→HINDPETRO, ZOMATO→ETERNAL, "
                "CANARABANK→CANBK, DALMIACEM→DALBHARAT. Analyzing on retired tickers yields "
                "wrong/empty quotes — always remap first."
            ),
            "tags": [
                "symbols", "nse", "rename", "demerger", "eternal", "hindpetro",
                "adaniensol", "education",
            ],
            "source": "bysel_builtin_v14",
        },
        # ── Personal finance / Mutual funds (Varsity Module) ─────────
        {
            "id": "pf_tvm_retirement",
            "title": "Personal finance: TVM, SIP, retirement corpus",
            "content": (
                "Time value of money: FV=PV×(1+r)^n; PV=FV/(1+r)^n; years to double≈72/rate%. "
                "Real return=(1+nominal)/(1+inflation)−1. SIP FV uses the ordinary annuity form. "
                "Educational retirement corpus≈annual expense/withdrawal rate (e.g. 4%). "
                "Back-solve the SIP needed for that corpus. Build an emergency fund of ~3–12 months "
                "essentials before aggressive equity SIPs. Not personalized advice."
            ),
            "tags": [
                "personal finance", "tvm", "sip", "retirement", "rule of 72",
                "real return", "emergency fund", "education",
            ],
            "source": "bysel_builtin_v12",
        },
        {
            "id": "pf_mutual_funds_core",
            "title": "Mutual funds: NAV, categories, TER, metrics",
            "content": (
                "Mutual funds pool capital under a SEBI mandate; you hold units at NAV. "
                "Equity categories span large/mid/small/flexi/ELSS; debt funds face rate and credit risk. "
                "Index funds/ETFs track indexes cheaply; arbitrage funds seek hedged cash–futures gaps. "
                "Judge with rolling returns, SD/beta/Sharpe/Sortino/capture, and TER — direct plans "
                "usually cost less than regular. Read the factsheet for mandate, loads, and overlap. "
                "Asset allocation by goal horizon beats chasing last year’s star fund."
            ),
            "tags": [
                "mutual fund", "nav", "expense ratio", "ter", "index fund", "etf",
                "rolling returns", "asset allocation", "smart beta", "education",
            ],
            "source": "bysel_builtin_v12",
        },
        {
            "id": "pf_review_checklist",
            "title": "Personal finance review: goals, debt, insurance buffer",
            "content": (
                "Review order: goals → cashflow → high-interest debt → term/health insurance adequacy → "
                "emergency fund → asset allocation → know-your-fund (mandate, TER, overlap) → rebalance. "
                "Insurance is protection, not an investment substitute. Educational checklist only."
            ),
            "tags": [
                "personal finance review", "financial planning", "emergency fund",
                "asset allocation", "know your fund", "education",
            ],
            "source": "bysel_builtin_v12",
        },
        # ── Trading systems (Varsity Module 10) ──────────────────────
        {
            "id": "tsys_pair_trading",
            "title": "Pair trading systems (correlation + regression)",
            "content": (
                "A trading system is a quantified process, not gut tips. Pair trading fades the "
                "relationship between two related stocks (ratio/spread). Method 1: high correlation + "
                "ratio z-score / density cues (often |z|≳2). Method 2: regress A on B for hedge ratio, "
                "fade residual extremes only if residuals look stationary (ADF). Correlation alone is "
                "not cointegration. Paper costs and hard stops."
            ),
            "tags": [
                "trading system", "pair trading", "z-score", "density curve", "adf",
                "cointegration", "regression", "education",
            ],
            "source": "bysel_builtin_v11",
        },
        {
            "id": "tsys_momentum_calendar",
            "title": "Momentum portfolios and calendar spread systems",
            "content": (
                "Momentum is rate-of-change of returns (prefer % ROC). Rank a universe, hold leaders, "
                "rebalance on a schedule with risk controls. Calendar spreads trade same underlier "
                "across expiries — mostly spread risk, small P&L, cost-sensitive; signals often near "
                "expiry. Backtest both legs' edge before scaling leverage."
            ),
            "tags": [
                "momentum", "momentum portfolio", "calendar spread", "trading system",
                "education", "futures",
            ],
            "source": "bysel_builtin_v11",
        },
        # ── Risk management (Varsity Module 9) ───────────────────────
        {
            "id": "risk_mgmt_core",
            "title": "Risk management: portfolio risk, VaR, sizing",
            "content": (
                "Diversify to cut unsystematic risk; hedge systematic risk. Portfolio expected return "
                "is weight×return sum. Correlation/covariance drive portfolio variance. "
                "Parametric VaR≈Value×z×σ×√t (z≈1.65 for ~95%). Position size with % risk: "
                "qty≈(equity×risk%)/|entry−stop|. Recovery after large losses is nonlinear — "
                "survive first. Use quarter-Kelly caps; journal to fight biases."
            ),
            "tags": [
                "risk management", "var", "value at risk", "position sizing", "kelly",
                "portfolio variance", "drawdown", "education",
            ],
            "source": "bysel_builtin_v10",
        },
        {
            "id": "risk_mgmt_psychology",
            "title": "Trading biases and recovery trauma",
            "content": (
                "Gambler's fallacy, anchoring, recency, confirmation, and hindsight bias distort size "
                "and exits. After −50% you need +100% to recover — overbetting small accounts creates "
                "recovery trauma. Antidote: written risk %, stops, and consistent sizing independent "
                "of the last streak."
            ),
            "tags": [
                "trading biases", "psychology", "anchoring", "gamblers fallacy",
                "recovery trauma", "education", "risk",
            ],
            "source": "bysel_builtin_v10",
        },
        # ── Currency / Commodity / G-Sec (Varsity Module 8) ──────────
        {
            "id": "ccg_currency_fx",
            "title": "Currency pairs, USDINR, IRP",
            "content": (
                "FX trades as Base/Quote (USDINR=83 means $1 costs ₹83). Indian exchange currency "
                "derivatives focus on INR pairs; global FX is near-24×5. USDINR reacts to rate "
                "differentials, flows, crude, and RBI. Forward intuition: "
                "F≈S×(1+r_INR×T)/(1+r_USD×T). Events move pairs — size for gaps."
            ),
            "tags": [
                "currency", "usdinr", "forex", "interest rate parity", "fx", "education", "fno",
            ],
            "source": "bysel_builtin_v9",
        },
        {
            "id": "ccg_commodities",
            "title": "MCX commodities: gold, crude, metals, gas",
            "content": (
                "MCX hosts metals/energy; NCDEX is agri-heavy. Know quote unit, lot, tick, expiry, "
                "delivery. P&L/tick=(lot/quote unit)×tick; contract value=(price×lot)/quote unit. "
                "Gold tracks $/oz×USDINR×duties; prefer liquid near-month. Crude/gas are volatile; "
                "base metals track growth/inventories. Square off before unwanted physical delivery."
            ),
            "tags": [
                "commodity", "mcx", "ncdex", "gold", "silver", "crude", "natural gas",
                "copper", "education",
            ],
            "source": "bysel_builtin_v9",
        },
        {
            "id": "ccg_gsec",
            "title": "G-Secs, T-bills, SDLs",
            "content": (
                "T-bills: discount instruments ≤1y; yield≈(discount/price)×(365/days). Dated G-Secs "
                "pay semi-annual coupons (e.g. 740GS2035≈7.40% to 2035). SDLs are state loans with "
                "similar mechanics. Prices fall when yields rise. Retail access via RBI/NSE pathways; "
                "tax treatment differs for interest vs gains — verify current rules."
            ),
            "tags": [
                "gsec", "g-sec", "treasury bill", "t-bill", "sdl", "bond yield", "ytm",
                "education", "rbi",
            ],
            "source": "bysel_builtin_v9",
        },
        # ── Option theory literacy (Varsity Module 5) ────────────────
        {
            "id": "opt_theory_basics",
            "title": "Call/put basics, premium, moneyness",
            "content": (
                "Call buyer has right to buy at strike; put buyer has right to sell. Premium = "
                "intrinsic + time value. Call intrinsic=max(0,S−K); put=max(0,K−S). ITM/ATM/OTM "
                "describe moneyness. Long call/put max loss = premium; short options earn premium "
                "with large adverse risk and margin. Expiry BE: call K+prem, put K−prem."
            ),
            "tags": [
                "option theory", "call option", "put option", "premium", "moneyness",
                "intrinsic", "itm", "atm", "otm", "options", "fno", "education",
            ],
            "source": "bysel_builtin_v8",
        },
        {
            "id": "opt_theory_greeks",
            "title": "Option Greeks and volatility",
            "content": (
                "Delta≈premium change per ₹1 spot; Gamma=delta change; Theta≈daily time decay; "
                "Vega≈premium change per 1 vol point. HV annualises past return stdev (×√252); "
                "IV is volatility priced by the market. Greeks interact — especially ATM near expiry. "
                "BYSEL can show educational BS Greeks using HV when live IV is unavailable."
            ),
            "tags": [
                "greeks", "delta", "gamma", "theta", "vega", "iv", "historical volatility",
                "options", "fno", "education",
            ],
            "source": "bysel_builtin_v8",
        },
        {
            "id": "opt_theory_pnl_settlement",
            "title": "Options P&L, M2M, physical settlement note",
            "content": (
                "Buyer/seller P&L are mirrors. Track premium MTM before expiry; at expiry premium "
                "tends to intrinsic. Stock F&O may involve physical settlement obligations — "
                "square off or know delivery rules. Index options are typically cash-settled. "
                "Educational only; confirm NSE/broker processes."
            ),
            "tags": [
                "options m2m", "pnl", "physical settlement", "expiry", "options", "fno",
                "education",
            ],
            "source": "bysel_builtin_v8",
        },
        # ── Option strategies literacy (Varsity Module 6) ────────────
        {
            "id": "opt_strat_map",
            "title": "Option strategies map (India F&O)",
            "content": (
                "Match view to structure: moderate bull → bull call/put spread; strong bull → "
                "call ratio back spread; moderate bear → bear put/call spread; strong bear → "
                "put ratio back spread; big move unsure → long straddle/strangle; range → "
                "short strangle/iron condor. Always define max loss, breakevens, and margin first."
            ),
            "tags": [
                "option strategies", "options", "fno", "straddle", "iron condor", "spread",
                "education",
            ],
            "source": "bysel_builtin_v7",
        },
        {
            "id": "opt_vertical_spreads",
            "title": "Vertical debit and credit spreads",
            "content": (
                "Bull call: buy lower call, sell higher call; max loss=debit; max profit=width−debit; "
                "BE=lower+debit. Bull put: sell higher put, buy lower put; max profit=credit; "
                "max loss=width−credit; BE=higher−credit. Bear put/call are the bearish mirrors. "
                "Use for moderate views; you cap both risk and reward."
            ),
            "tags": [
                "bull call spread", "bull put spread", "bear put spread", "bear call spread",
                "options", "fno", "education",
            ],
            "source": "bysel_builtin_v7",
        },
        {
            "id": "opt_vol_range_strats",
            "title": "Straddle, strangle, iron condor, PCR, max pain",
            "content": (
                "Long straddle/strangle need a move larger than premium (watch IV crush). "
                "Short straddle/strangle and iron condor prefer range/vol crush — short naked "
                "has large tails; iron condor defines risk. PCR = put/call OI or volume. "
                "Max pain is a writer-pain heuristic from the chain, not a pin guarantee."
            ),
            "tags": [
                "straddle", "strangle", "iron condor", "max pain", "pcr", "put call ratio",
                "options", "fno", "education",
            ],
            "source": "bysel_builtin_v7",
        },
        {
            "id": "opt_ratio_synthetic",
            "title": "Ratio back spreads and synthetic long",
            "content": (
                "Call ratio back spread (1×2) suits strong bullish views: sell 1 lower call, buy 2 "
                "higher; max-loss pocket near higher strike, large upside beyond upper BE. "
                "Put ratio is the bearish mirror. Synthetic long = long call + short put ≈ futures "
                "bullishness; short put risk is real. Paper the payoff before any live margin."
            ),
            "tags": [
                "call ratio back spread", "put ratio back spread", "synthetic long",
                "bear call ladder", "options", "fno", "education",
            ],
            "source": "bysel_builtin_v7",
        },
        # ── Futures literacy (Varsity Module 4, paraphrased) ─────────
        {
            "id": "fut_forwards_vs_futures",
            "title": "Forwards vs futures (India)",
            "content": (
                "Forwards are OTC custom agreements with counterparty risk. Futures are exchange-"
                "standardised, cleared, margined and marked-to-market daily — the retail default on NSE. "
                "Lot size × futures price = contract value; you post margin, not full notional."
            ),
            "tags": ["futures", "forwards", "fno", "derivatives", "education", "nse"],
            "source": "bysel_builtin_v6",
        },
        {
            "id": "fut_leverage_mtm",
            "title": "Futures leverage, MTM and margin calls",
            "content": (
                "Leverage ≈ contract value / margin. Rough wipeout % ≈ 1/leverage. Daily MTM settles "
                "P&L in cash: (settle − prior ref) × lot (sign flips for shorts). Losses can trigger "
                "margin calls even if the longer thesis is intact. Prefer paper practice of lot risk."
            ),
            "tags": ["futures", "leverage", "mtm", "margin", "fno", "education", "risk"],
            "source": "bysel_builtin_v6",
        },
        {
            "id": "fut_pricing_carry",
            "title": "Futures pricing cost of carry",
            "content": (
                "Fair value F ≈ S×(1+Rf×T/365) − D. Basis = futures − spot. Premium/contango-like "
                "vs discount/backwardation-like. Market price can differ from fair value due to costs, "
                "dividends and positioning. Calendar spreads trade relative expiries of the same underlier."
            ),
            "tags": [
                "futures", "pricing", "cost of carry", "basis", "contango", "backwardation",
                "calendar spread", "fno", "education",
            ],
            "source": "bysel_builtin_v6",
        },
        {
            "id": "fut_shorting_impact",
            "title": "Shorting spot vs futures; impact cost",
            "content": (
                "Spot shorts in India are typically intraday — overnight short delivery risks auction "
                "penalty. Futures shorts can be carried with margin + MTM. Impact cost ≈ round-trip "
                "bid/ask loss vs mid; Nifty futures usually far more liquid than thin single stocks."
            ),
            "tags": [
                "shorting", "futures", "impact cost", "liquidity", "nifty", "fno", "education",
            ],
            "source": "bysel_builtin_v6",
        },
        {
            "id": "fut_hedge_oi",
            "title": "Futures hedging, beta lots, OI reading",
            "content": (
                "Diversify to cut unsystematic risk; hedge systematic risk with index futures. "
                "Hedge value ≈ portfolio β × portfolio value; lots ≈ hedge value / (Nifty F × lot). "
                "Whole lots force under/over-hedge. OI is open contracts (not volume). Price↑OI↑ new "
                "longs; Price↑OI↓ short covering; Price↓OI↑ new shorts; Price↓OI↓ long liquidation. "
                "Stock F&O may involve physical settlement near expiry — confirm NSE rules."
            ),
            "tags": [
                "hedging", "beta", "open interest", "oi", "physical settlement", "futures",
                "nifty", "fno", "education",
            ],
            "source": "bysel_builtin_v6",
        },
        # ── SEBI educational disclaimer ──────────────────────────────
        {
            "id": "v2_sebi_disclaimer",
            "title": "SEBI educational disclaimer (BYSEL)",
            "content": (
                "BYSEL and its AI assistants are for education and paper/simulation practice. "
                "They are NOT a SEBI-registered Research Analyst (RA), Investment Adviser (IA), "
                "or portfolio manager. Nothing here is a buy/sell tip, guarantee, or personalized "
                "advisory under SEBI regulations. Markets involve loss of capital. Verify with "
                "live NSE/BSE data and consult a SEBI-registered intermediary before real trades."
            ),
            "tags": ["sebi", "disclaimer", "compliance", "education", "ra", "advice", "risk"],
            "source": "bysel_builtin_v2",
        },
    ]
    items = [KnowledgeItem(**item) for item in raw]
    items.extend(nse_official_literacy_items())
    return items

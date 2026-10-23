"""NQ/MNQ quarterly roll calendar and continuous series construction.

Contracts: H (Mar), M (Jun), U (Sep), Z (Dec); cash-settled on the third Friday.

Front month
-----------
* ``volume`` method: the front month switches when the next contract's *daily* volume
  exceeds the current one's. Day D's volume is only known after D closes, so the
  switch takes effect from session D+1 (no lookahead).
* ``cme_convention`` fallback: roll on the session 8 calendar days before expiry
  (the Thursday of the prior week).

Adjusted series
---------------
Trading is simulated on **unadjusted** front-month prices. Long-lookback indicators use
an additively adjusted series. Conventional back-adjustment anchors to the *latest*
contract, which makes every historical value depend on roll gaps that happen later; a
truncated history would then produce different feature values (a lookahead leak the
truncation property test catches). nqlab therefore anchors the adjustment at the
*earliest* contract and applies each roll gap only from the roll onward. Point
differences are preserved exactly, which is all MACD/EMA-difference logic needs, and the
series is causal. ``adj_offset = close_badj - close`` converts back to tradable prices.
"""

from __future__ import annotations

import datetime as dt
import re

import polars as pl

MONTH_CODES = {"H": 3, "M": 6, "U": 9, "Z": 12}
CODE_FOR_MONTH = {v: k for k, v in MONTH_CODES.items()}
_CONTRACT_RE = re.compile(r"^(?P<root>M?NQ)(?P<code>[HMUZ])(?P<yy>\d{1,2})$")


def third_friday(year: int, month: int) -> dt.date:
    d = dt.date(year, month, 15)
    while d.weekday() != 4:
        d += dt.timedelta(days=1)
    return d


def parse_contract(symbol: str, ref_year: int | None = None) -> tuple[str, int, int]:
    """'NQZ25' -> ('NQ', 2025, 12). One-digit years resolve to the decade nearest
    ``ref_year`` (default: current year)."""
    m = _CONTRACT_RE.match(symbol.upper())
    if not m:
        raise ValueError(f"not a quarterly NQ/MNQ contract: {symbol!r}")
    yy = m.group("yy")
    if len(yy) == 2:
        year = 2000 + int(yy)
    else:
        ref = ref_year or dt.date.today().year
        base = ref - ref % 10
        cands = [base - 10 + int(yy), base + int(yy), base + 10 + int(yy)]
        year = min(cands, key=lambda y: abs(y - ref))
    return m.group("root"), year, MONTH_CODES[m.group("code")]


def contract_symbol(root: str, year: int, month: int) -> str:
    return f"{root}{CODE_FOR_MONTH[month]}{year % 100:02d}"


def expiry(symbol: str) -> dt.date:
    _, y, m = parse_contract(symbol)
    return third_friday(y, m)


def quarterly_contracts(root: str, start: dt.date, end: dt.date) -> list[str]:
    out = []
    y, m = start.year, start.month
    while (y, m) <= (end.year + 1, 12):
        if m in CODE_FOR_MONTH:
            out.append(contract_symbol(root, y, m))
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out


def cme_convention_front(session_dates: list[dt.date], root: str, days_before: int = 8) -> pl.DataFrame:
    """Front contract per session using the CME roll-date convention."""
    if not session_dates:
        return pl.DataFrame(schema={"session_date": pl.Date, "front": pl.Utf8})
    contracts = quarterly_contracts(root, min(session_dates) - dt.timedelta(days=100), max(session_dates))
    rolls = [(expiry(c) - dt.timedelta(days=days_before), c) for c in contracts]
    rows = []
    for d in sorted(session_dates):
        front = next(c for roll_day, c in rolls if d < roll_day)
        rows.append((d, front))
    return pl.DataFrame(rows, schema={"session_date": pl.Date, "front": pl.Utf8}, orient="row")


def volume_front(daily_volume: pl.DataFrame, root: str) -> pl.DataFrame:
    """Front month by volume. ``daily_volume``: session_date, symbol, volume.

    Monotone: never rolls back to an earlier contract, and never holds an expired one.
    The decision made on day D's volume applies from D+1.
    """
    dv = daily_volume.sort("session_date")
    dates = dv["session_date"].unique().sort().to_list()
    if not dates:
        return pl.DataFrame(schema={"session_date": pl.Date, "front": pl.Utf8})
    vol = {(r["session_date"], r["symbol"]): r["volume"] for r in dv.iter_rows(named=True)}
    syms = sorted({s for _, s in vol}, key=lambda s: parse_contract(s)[1:])
    conv = cme_convention_front(dates, root)
    conv_map = dict(conv.iter_rows())
    # start on the conventional front for the first day (no volume history yet)
    current = conv_map[dates[0]] if conv_map[dates[0]] in syms else syms[0]
    rows = []
    for d in dates:
        # expired contracts cannot be traded
        while expiry(current) <= d and syms.index(current) + 1 < len(syms):
            current = syms[syms.index(current) + 1]
        rows.append((d, current))
        i = syms.index(current)
        if i + 1 < len(syms):
            nxt = syms[i + 1]
            if vol.get((d, nxt), 0) > vol.get((d, current), 0):
                current = nxt  # effective from the next session
    return pl.DataFrame(rows, schema={"session_date": pl.Date, "front": pl.Utf8}, orient="row")


def build_continuous(bars: pl.DataFrame, front: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Assemble the front-month series with causal adjusted prices.

    ``bars`` must carry session_date and symbol. Returns (series, rolls) where series has
    unadjusted OHLC plus ``*_badj`` (additive) and ``*_radj`` (ratio) columns and
    ``adj_offset``; rolls lists each roll with its measured gap.
    """
    joined = bars.join(front, on="session_date", how="inner").filter(pl.col("symbol") == pl.col("front"))
    joined = joined.drop("front").sort("ts_utc")
    fr = front.sort("session_date")
    roll_rows = []
    prev_sym = None
    prev_date = None
    for d, sym in fr.iter_rows():
        if prev_sym is not None and sym != prev_sym:
            # gap measured at the last bar of the previous session where both traded
            a = bars.filter((pl.col("session_date") == prev_date) & (pl.col("symbol") == prev_sym)).select("ts_utc", "close")
            b = bars.filter((pl.col("session_date") == prev_date) & (pl.col("symbol") == sym)).select("ts_utc", "close")
            both = a.join(b, on="ts_utc", suffix="_new").sort("ts_utc")
            if both.height:
                last = both.row(-1, named=True)
                gap, ratio = last["close_new"] - last["close"], last["close_new"] / last["close"]
                method = "measured"
            else:
                gap, ratio, method = 0.0, 1.0, "unmeasured"
            roll_rows.append({"roll_session": d, "from": prev_sym, "to": sym, "gap": gap,
                              "ratio": ratio, "gap_method": method})
        prev_sym, prev_date = sym, d
    rolls = pl.DataFrame(roll_rows, schema={"roll_session": pl.Date, "from": pl.Utf8, "to": pl.Utf8,
                                            "gap": pl.Float64, "ratio": pl.Float64, "gap_method": pl.Utf8})
    return apply_adjustment(joined, rolls), rolls


def apply_adjustment(series: pl.DataFrame, rolls: pl.DataFrame) -> pl.DataFrame:
    """Add causal *_badj / *_radj columns given a roll table (empty -> identity)."""
    if rolls.height:
        cum = rolls.sort("roll_session").select(
            pl.col("roll_session").alias("session_date"),
            pl.col("gap").cum_sum().alias("cum_gap"),
            pl.col("ratio").cum_prod().alias("cum_ratio"),
        )
        series = series.sort("session_date").join_asof(cum, on="session_date", strategy="backward")
        series = series.with_columns(pl.col("cum_gap").fill_null(0.0), pl.col("cum_ratio").fill_null(1.0))
        series = series.sort("ts_utc")
    else:
        series = series.with_columns(pl.lit(0.0).alias("cum_gap"), pl.lit(1.0).alias("cum_ratio"))
    out = series.with_columns(
        *[(pl.col(c) - pl.col("cum_gap")).alias(f"{c}_badj") for c in ("open", "high", "low", "close")],
        *[(pl.col(c) / pl.col("cum_ratio")).alias(f"{c}_radj") for c in ("open", "high", "low", "close")],
    ).with_columns((pl.col("close_badj") - pl.col("close")).alias("adj_offset"))
    return out.drop("cum_gap", "cum_ratio")


def canonical_contract(symbol: str, first_seen_year: int) -> str | None:
    """'NQZ5' first seen in 2015 -> 'NQZ15'. Contracts are listed at most ~2 years before
    expiry, so the expiry year is the first year >= first_seen_year with that last digit.
    Returns None for spreads / non-quarterly symbols."""
    m = _CONTRACT_RE.match(symbol.upper())
    if not m:
        return None
    yy = m.group("yy")
    if len(yy) == 2:
        return symbol.upper()
    d = int(yy)
    year = first_seen_year + ((d - first_seen_year % 10) % 10)
    return contract_symbol(m.group("root"), year, MONTH_CODES[m.group("code")])

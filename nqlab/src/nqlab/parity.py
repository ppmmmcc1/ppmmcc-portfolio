"""Discretionary parity (spec 13.3).

``label``   the operator hand-labels sessions ("I would have taken this trade here");
``compare`` precision / recall of the coded signals against those labels, with a diff of
            every divergence: this catches "the code isn't testing what I actually do";
``import``  TopstepX trade history vs model trades: fill rate, slippage and expectancy
            differences with confidence intervals. Once live, the only thing that catches
            a fill model that is quietly lying.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

ET = ZoneInfo("America/New_York")
LABEL_COLS = {"session_date": pl.Date, "time_et": pl.Utf8, "side": pl.Utf8, "note": pl.Utf8}

# TopstepX "Trades" export headers (configurable: pass a mapping to import_topstepx)
TOPSTEPX_COLUMNS = {"entry_ts": "EnteredAt", "exit_ts": "ExitedAt", "entry_price": "EntryPrice",
                    "exit_price": "ExitPrice", "fees": "Fees", "pnl": "PnL", "size": "Size", "side": "Type",
                    "contract": "ContractName"}


def labels_path(settings) -> Path:
    return settings.path("research_log") / "parity_labels.csv"


def add_label(settings, session_date: dt.date, time_et: str, side: str, note: str = "") -> Path:
    side = side.lower()
    if side not in ("long", "short", "none"):
        raise ValueError("side must be long, short or none (none = 'I would NOT trade this session')")
    dt.datetime.strptime(time_et, "%H:%M")
    p = labels_path(settings)
    row = pl.DataFrame({"session_date": [session_date], "time_et": [time_et], "side": [side], "note": [note]}, schema=LABEL_COLS)
    if p.exists():
        row = pl.concat([pl.read_csv(p, schema=LABEL_COLS), row])
    p.parent.mkdir(parents=True, exist_ok=True)
    row.write_csv(p)
    return p


def _et_minutes(ts: dt.datetime) -> int:
    t = ts.astimezone(ET)
    return t.hour * 60 + t.minute


def compare_labels(labels: pl.DataFrame, trades: pl.DataFrame, tolerance_min: int = 5) -> dict[str, Any]:
    """Match coded entries to labels on (session, side, |time| <= tolerance)."""
    lab_sessions = set(labels["session_date"].to_list())
    lab = labels.filter(pl.col("side") != "none")
    tr = trades.filter(pl.col("session_date").is_in(list(lab_sessions)))
    coded = [(r["session_date"], "long" if r["side"] > 0 else "short", _et_minutes(r["entry_ts"])) for r in tr.iter_rows(named=True)]
    human = [(r["session_date"], r["side"], int(r["time_et"][:2]) * 60 + int(r["time_et"][3:])) for r in lab.iter_rows(named=True)]
    used = set()
    matches, only_code = [], []
    for c in coded:
        hit = next((i for i, h in enumerate(human) if i not in used and h[0] == c[0] and h[1] == c[1]
                    and abs(h[2] - c[2]) <= tolerance_min), None)
        if hit is None:
            only_code.append(c)
        else:
            used.add(hit)
            matches.append((c, human[hit]))
    only_human = [h for i, h in enumerate(human) if i not in used]
    tp = len(matches)
    fmt = lambda x: {"date": str(x[0]), "side": x[1], "time_et": f"{x[2] // 60:02d}:{x[2] % 60:02d}"}  # noqa: E731
    return {
        "labelled_sessions": len(lab_sessions), "human_trades": len(human), "coded_trades": len(coded),
        "precision": tp / len(coded) if coded else float("nan"),
        "recall": tp / len(human) if human else float("nan"),
        "coded_not_human": [fmt(x) for x in only_code], "human_not_coded": [fmt(x) for x in only_human],
        "decision_required": bool(only_code or only_human),
    }


def import_topstepx(path: Path, mapping: dict[str, str] | None = None, tz: str = "America/New_York") -> pl.DataFrame:
    m = {**TOPSTEPX_COLUMNS, **(mapping or {})}
    raw = pl.read_csv(path, infer_schema_length=10000)
    lower = {c.lower(): c for c in raw.columns}

    def col(k):
        name = m[k]
        if name in raw.columns:
            return raw[name]
        if name.lower() in lower:
            return raw[lower[name.lower()]]
        raise ValueError(f"TopstepX CSV has no column for {k!r} (expected {name!r}); headers: {raw.columns}")

    def ts(s: pl.Series) -> pl.Series:
        s = s.cast(pl.Utf8).str.strip_chars()
        p = s.str.to_datetime(time_unit="us", strict=False)
        if p.dtype.time_zone is None:  # type: ignore[union-attr]
            p = p.dt.replace_time_zone(tz, ambiguous="earliest")
        return p.dt.convert_time_zone("UTC")

    side = col("side").cast(pl.Utf8).str.to_lowercase()
    out = pl.DataFrame({
        "entry_ts": ts(col("entry_ts")), "exit_ts": ts(col("exit_ts")),
        "side": pl.Series(np.where(side.str.contains("long|buy").to_numpy(), 1, -1)),
        "entry_price": col("entry_price").cast(pl.Float64), "exit_price": col("exit_price").cast(pl.Float64),
        "fees": col("fees").cast(pl.Float64), "pnl": col("pnl").cast(pl.Float64), "size": col("size").cast(pl.Int64),
    })
    from nqlab.calendar.sessions import session_date_expr

    return out.with_columns(session_date_expr("entry_ts").alias("session_date"))


def compare_fills(live: pl.DataFrame, model: pl.DataFrame, tolerance_min: int = 5, n_boot: int = 5000,
                  seed: int = 0) -> dict[str, Any]:
    """Pair live and model trades (same session, side, entry within tolerance)."""
    mt = model.with_columns(pl.col("entry_ts").dt.epoch("s").alias("_t"))
    lt = live.with_columns(pl.col("entry_ts").dt.epoch("s").alias("_t"))
    pairs, used = [], set()
    for r in lt.iter_rows(named=True):
        cand = mt.filter((pl.col("session_date") == r["session_date"]) & (pl.col("side") == r["side"])
                         & ((pl.col("_t") - r["_t"]).abs() <= tolerance_min * 60))
        cand = cand.with_row_index("_k")
        best = None
        for c in cand.iter_rows(named=True):
            key = (c["session_date"], c["_t"])
            if key not in used:
                best = c
                used.add(key)
                break
        if best is not None:
            pairs.append((r, best))
    model_in_window = mt.filter(pl.col("session_date").is_in(lt["session_date"].unique().implode()))
    n_model = model_in_window.height
    slip = np.array([(lv["entry_price"] - md["entry_price"]) * lv["side"] for lv, md in pairs])  # + = live worse
    live_pnl_pc = np.array([lv["pnl"] / max(lv["size"], 1) for lv, _ in pairs])
    model_pnl_pc = np.array([md["pnl_usd"] / max(md["contracts"], 1) for _, md in pairs])
    rng = np.random.default_rng(seed)

    def ci(x):
        if len(x) < 2:
            return (float("nan"), float("nan"))
        b = rng.choice(x, (n_boot, len(x))).mean(axis=1)
        return (float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975)))

    diff = live_pnl_pc - model_pnl_pc
    return {
        "live_trades": live.height, "model_trades_same_sessions": n_model, "paired": len(pairs),
        "model_fill_rate_realized": len(pairs) / n_model if n_model else float("nan"),
        "live_trades_model_missed": live.height - len(pairs),
        "entry_slippage_pts_mean": float(slip.mean()) if len(slip) else float("nan"), "entry_slippage_pts_ci": ci(slip),
        "expectancy_live_minus_model_usd": float(diff.mean()) if len(diff) else float("nan"),
        "expectancy_diff_ci": ci(diff),
    }

"""Feature registry. Every feature must declare which price series it reads.

All features are pure functions of bar history ending at or before the decision bar's
close: no centered windows, no backward fills from the future. The truncation property
test (tests/test_lookahead.py) runs every registered feature.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import polars as pl

SERIES = ("unadjusted", "back_adjusted", "ratio_adjusted")
_SUFFIX = {"unadjusted": "", "back_adjusted": "_badj", "ratio_adjusted": "_radj"}


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    fn: Callable[..., pl.DataFrame]
    series: str
    defaults: dict[str, Any] = field(default_factory=dict)
    approximate_note: str | None = None


FEATURES: dict[str, FeatureSpec] = {}


def feature(name: str, *, series: str | None, defaults: dict[str, Any] | None = None,
            approximate_note: str | None = None, unadjusted_only: bool = False):
    """Register a feature. ``series`` is mandatory; ``"param"`` means the caller picks it
    via a ``series=`` argument (validated at call time)."""
    if series is None:
        raise TypeError(f"feature {name!r} must declare its price series")
    if series not in SERIES and series != "param":
        raise ValueError(f"feature {name!r}: unknown series {series!r}")
    if unadjusted_only and series != "unadjusted":
        raise ValueError(f"feature {name!r} may only use unadjusted prices")

    def deco(fn):
        FEATURES[name] = FeatureSpec(name, fn, series, dict(defaults or {}), approximate_note)
        return fn

    return deco


def col(source: str, series: str) -> pl.Expr:
    """Price expression for ``source`` (open/high/low/close/hlc3/ohlc4) in ``series``."""
    if series not in SERIES:
        raise ValueError(f"unknown series {series!r}")
    sfx = _SUFFIX[series]
    if source in ("open", "high", "low", "close"):
        return pl.col(source + sfx)
    if source == "hlc3":
        return (pl.col("high" + sfx) + pl.col("low" + sfx) + pl.col("close" + sfx)) / 3.0
    if source == "ohlc4":
        return (pl.col("open" + sfx) + pl.col("high" + sfx) + pl.col("low" + sfx) + pl.col("close" + sfx)) / 4.0
    raise ValueError(f"unknown price source {source!r}")


def to_unadjusted(values: pl.Expr, series: str) -> pl.Expr:
    """Convert a price-level feature computed in ``series`` space to tradable prices at
    the current bar. Additive: x - adj_offset. Ratio: x * cum_ratio (= close/close_radj)."""
    if series == "unadjusted":
        return values
    if series == "back_adjusted":
        return values - pl.col("adj_offset")
    return values * (pl.col("close") / pl.col("close_radj"))


def compute(md, name: str, **params) -> pl.DataFrame:
    spec = FEATURES[name]
    p = {**spec.defaults, **params}
    if spec.series != "param":
        p.pop("series", None)
    elif p.get("series") not in SERIES:
        raise ValueError(f"{name}: series= must be one of {SERIES}")
    cache = md.meta.setdefault("_feature_cache", {})
    key = (name, tuple(sorted((k, str(v)) for k, v in p.items())))
    if key not in cache:
        out = spec.fn(md, **p)
        if out.height != md.bars.height:
            raise AssertionError(f"feature {name} returned {out.height} rows for {md.bars.height} bars")
        cache[key] = out
    return cache[key]

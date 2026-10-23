"""Strategy spec files (specs/*.yaml): load, flatten, find undecided fields, grid dims."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import yaml

from nqlab.hashing import stable_hash

UNDECIDED = "REQUIRED_UNDECIDED"
META_KEYS = ("spec_version", "strategy", "variant")


class SpecIncomplete(ValueError):  # also raised for runs; see cli._guard
    def __init__(self, missing: list[str]):
        self.missing = missing
        super().__init__(
            "spec has undecided fields; nqlab refuses to run it until they are filled:\n  - "
            + "\n  - ".join(missing)
        )


def load_spec(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def undecided(spec: dict[str, Any]) -> list[str]:
    return sorted(k for k, v in flatten(spec).items() if v == UNDECIDED
                  or (isinstance(v, list) and UNDECIDED in v))


def spec_hash(spec: dict[str, Any]) -> str:
    return stable_hash(spec)


# leaves whose VALUE is naturally a list (not a grid dimension)
LIST_VALUED = {"volume_profile.long_levels", "volume_profile.short_levels", "trigger.timeframes"}


def grid_dimensions(spec: dict[str, Any]) -> dict[str, list[Any]]:
    """Leaves given as lists are grid dimensions (except list-valued leaves, which are
    dimensions only when given as a list of lists)."""
    dims = {}
    for k, v in flatten(spec).items():
        if k in META_KEYS:
            continue
        if isinstance(v, list):
            if k in LIST_VALUED:
                if v and all(isinstance(x, list) for x in v):
                    dims[k] = v
            else:
                dims[k] = v
    return dims


def expand(spec: dict[str, Any]) -> list[dict[str, Any]]:
    flat = {k: v for k, v in flatten(spec).items()}
    dims = grid_dimensions(spec)
    if not dims:
        return [flat]
    keys = list(dims)
    out = []
    for combo in itertools.product(*(dims[k] for k in keys)):
        p = dict(flat)
        p.update(dict(zip(keys, combo)))
        out.append(p)
    return out


# Canonical field list (spec section 13.1). A spec that OMITS any of these is as incomplete
# as one that leaves it REQUIRED_UNDECIDED: deleting a line must not create a silent default.
REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {"bb_vp_macd": (
    "variant",
    "context.direction_definition", "context.agreement_rule", "context.hourly_alignment",
    "context.macd_15m.fast", "context.macd_15m.slow", "context.macd_15m.signal", "context.macd_15m.source",
    "context.macd_1h.fast", "context.macd_1h.slow", "context.macd_1h.signal", "context.macd_1h.source",
    "bollinger.timeframe", "bollinger.length", "bollinger.mult", "bollinger.basis", "bollinger.stdev",
    "bollinger.source", "bollinger.reprice_each_bar",
    "volume_profile.source", "volume_profile.composite_days", "volume_profile.value_area_pct",
    "volume_profile.bin_size", "volume_profile.poc_tie", "volume_profile.build_from", "volume_profile.align_ticks",
    "volume_profile.long_levels", "volume_profile.short_levels", "volume_profile.order_price",
    "trigger.timeframes", "trigger.both_within_bars", "trigger.cross_definition", "trigger.min_hist_magnitude",
    "trigger.confirmation_window_bars", "trigger.extreme_lookback_bars",
    "trigger.macd.fast", "trigger.macd.slow", "trigger.macd.signal", "trigger.macd.source",
    "orders.window_start_et", "orders.window_end_et", "orders.expiry_minutes", "orders.cancel_if_band_moves_ticks",
    "orders.max_concurrent_positions", "orders.max_trades_per_day", "orders.stop_after_first_win",
    "orders.stop_after_first_loss", "orders.stop.type", "orders.stop.value", "orders.stop.buffer_ticks",
    "orders.target.type", "orders.target.value", "orders.partials", "orders.breakeven.enabled",
    "orders.breakeven.trigger_r", "orders.time_stop_et", "orders.skip_days.fomc", "orders.skip_days.release_0830",
    "orders.skip_days.half_days", "orders.skip_days.roll_week",
    "sizing.instrument", "sizing.risk_per_trade_usd", "sizing.risk_pct_of_mll_buffer", "sizing.daily_stop_usd",
)}

ALLOWED: dict[str, tuple] = {
    "variant": ("limit", "trigger"),
    "context.direction_definition": ("line_vs_signal", "hist_sign", "line_vs_zero", "line_slope"),
    "context.agreement_rule": ("both", "either", "1h_dominates"),
    "context.hourly_alignment": ("clock", "session"),
    "context.macd_15m.source": ("close", "hlc3"), "context.macd_1h.source": ("close", "hlc3"),
    "trigger.macd.source": ("close", "hlc3"),
    "bollinger.timeframe": ("1m", "5m"), "bollinger.basis": ("sma", "ema"),
    "bollinger.stdev": ("population", "sample"), "bollinger.source": ("close", "hlc3"),
    "bollinger.reprice_each_bar": (True, False),
    "volume_profile.source": ("prior_rth", "prior_globex", "composite", "developing"),
    "volume_profile.poc_tie": ("mid", "low", "high"), "volume_profile.build_from": ("trades", "bars"),
    "volume_profile.order_price": ("band", "level", "better"),
    "trigger.cross_definition": ("line_cross_signal", "hist_sign_change", "line_cross_zero"),
    "orders.stop.type": ("fixed_ticks", "atr_multiple", "beyond_level"),
    "orders.target.type": ("opposite_band", "basis", "poc", "fixed_r"),
    "sizing.instrument": ("NQ", "MNQ"),
}


def missing_fields(spec: dict[str, Any], family: str = "bb_vp_macd") -> list[str]:
    flat = flatten(spec)
    return sorted(k for k in REQUIRED_FIELDS.get(family, ()) if k not in flat)


def invalid_values(spec: dict[str, Any]) -> list[tuple[str, Any, tuple]]:
    out = []
    for k, v in flatten(spec).items():
        if k in ALLOWED and v != UNDECIDED:
            vals = v if isinstance(v, list) and k not in LIST_VALUED else [v]
            for x in vals:
                if x not in ALLOWED[k]:
                    out.append((k, x, ALLOWED[k]))
    return out

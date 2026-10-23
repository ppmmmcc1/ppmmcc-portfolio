"""Signal arrays a strategy hands to the engine, and the engine parameters."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from nqlab.engine.kernel import LIMIT, MARKET, NONE, STOP, STOPLIMIT

ORDER_TYPES = {"none": NONE, "market": MARKET, "limit": LIMIT, "stop": STOP, "stop_limit": STOPLIMIT}


@dataclass
class Signals:
    """All arrays have one entry per engine bar and are decided at that bar's CLOSE.

    Entry orders carry their bracket as distances in points from the fill price
    (NaN = no stop / no target). Re-sending an order on a later bar replaces the
    resting one (that is how "reprice each bar" is expressed)."""

    n: int
    l_type: np.ndarray = field(init=False)
    l_px: np.ndarray = field(init=False)
    l_lim: np.ndarray = field(init=False)
    l_stop: np.ndarray = field(init=False)
    l_tgt: np.ndarray = field(init=False)
    s_type: np.ndarray = field(init=False)
    s_px: np.ndarray = field(init=False)
    s_lim: np.ndarray = field(init=False)
    s_stop: np.ndarray = field(init=False)
    s_tgt: np.ndarray = field(init=False)
    cancel_l: np.ndarray = field(init=False)
    cancel_s: np.ndarray = field(init=False)
    exit_now: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        n = self.n
        for side in ("l", "s"):
            setattr(self, f"{side}_type", np.zeros(n, np.int64))
            for k in ("px", "lim", "stop", "tgt"):
                setattr(self, f"{side}_{k}", np.full(n, np.nan))
        self.cancel_l = np.zeros(n, np.bool_)
        self.cancel_s = np.zeros(n, np.bool_)
        self.exit_now = np.zeros(n, np.bool_)

    def place(self, mask: np.ndarray, side: int, otype: str, price=np.nan, stop_dist=np.nan,
              target_dist=np.nan, limit=np.nan) -> None:
        p = "l" if side > 0 else "s"
        t = ORDER_TYPES[otype]
        getattr(self, f"{p}_type")[mask] = t
        for k, v in (("px", price), ("stop", stop_dist), ("tgt", target_dist), ("lim", limit)):
            arr = getattr(self, f"{p}_{k}")
            arr[mask] = v[mask] if isinstance(v, np.ndarray) else v


@dataclass(frozen=True)
class EngineParams:
    fill_tier: int = 2
    expiry_bars: int = 0          # 0 = good for the session
    max_trades_per_day: int = 1
    stop_after_win: bool = False
    stop_after_loss: bool = False
    be_trigger_r: float = float("nan")
    be_offset_ticks: int = 0

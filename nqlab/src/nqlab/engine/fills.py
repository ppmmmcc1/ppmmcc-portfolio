"""The four fill models as interchangeable classes.

Tier 1 Touch           optimistic UPPER BOUND only, never the headline
Tier 2 TradeThrough    the DEFAULT and the headline: price must trade one tick through
Tier 3 QueueModel      needs trades data: estimated queue ahead + cancellation haircut
Tier 4 MBOReplay       needs Databento MBO: exact queue position; used to calibrate 2 and 3
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from nqlab.engine.bars import EngineBars
from nqlab.engine.orders import EngineParams, Signals
from nqlab.engine.simulate import simulate


class FillModelUnavailable(RuntimeError):
    pass


@dataclass
class FillModel:
    tier: int
    name: str
    requires: str | None = None
    headline: bool = False
    label: str = ""

    def run(self, eb: EngineBars, sig: Signals, ep: EngineParams, tick: float = 0.25, **data) -> pl.DataFrame:
        raise NotImplementedError


class TouchFill(FillModel):
    def __init__(self):
        super().__init__(1, "touch", None, False, "Tier 1 touch (optimistic upper bound)")

    def run(self, eb, sig, ep, tick=0.25, **data):
        return simulate(eb, sig, EngineParams(**{**ep.__dict__, "fill_tier": 1}), tick)


class TradeThroughFill(FillModel):
    def __init__(self):
        super().__init__(2, "trade_through", None, True, "Tier 2 trade-through (headline)")

    def run(self, eb, sig, ep, tick=0.25, **data):
        return simulate(eb, sig, EngineParams(**{**ep.__dict__, "fill_tier": 2}), tick)


class QueueFill(FillModel):
    def __init__(self, haircut: float = 0.3):
        super().__init__(3, "queue", "trades", False, "Tier 3 queue model (trades data)")
        self.haircut = haircut

    def run(self, eb, sig, ep, tick=0.25, trades: pl.DataFrame | None = None, **data):
        if trades is None:
            raise FillModelUnavailable("tier 3 needs trades data (Databento 'trades' schema)")
        from nqlab.engine.tick_replay import TradesQueue, replay

        return replay(eb, sig, ep, trades, TradesQueue(trades, self.haircut), tick)


class MBOReplayFill(FillModel):
    def __init__(self):
        super().__init__(4, "mbo", "mbo", False, "Tier 4 MBO replay (exact queue)")

    def run(self, eb, sig, ep, tick=0.25, trades: pl.DataFrame | None = None, mbo: pl.DataFrame | None = None, **data):
        if mbo is None or trades is None:
            raise FillModelUnavailable("tier 4 needs Databento MBO (and trades) data")
        from nqlab.engine.tick_replay import MBOQueue, replay

        return replay(eb, sig, ep, trades, MBOQueue(mbo), tick)


def available_models(has_trades: bool = False, has_mbo: bool = False, haircut: float = 0.3) -> list[FillModel]:
    out: list[FillModel] = [TradeThroughFill(), TouchFill()]
    if has_trades:
        out.append(QueueFill(haircut))
    if has_mbo and has_trades:
        out.append(MBOReplayFill())
    return out

"""Name -> strategy class. Workers resolve strategies here (Windows spawn safety)."""

from __future__ import annotations

from nqlab.strategies.base import Strategy

_REGISTRY: dict[str, type[Strategy]] = {}


def register(cls: type[Strategy]) -> type[Strategy]:
    _REGISTRY[cls.name] = cls
    return cls


def get_strategy(name: str) -> Strategy:
    _load()
    if name not in _REGISTRY:
        raise KeyError(f"unknown strategy {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]()


def names() -> list[str]:
    _load()
    return sorted(_REGISTRY)


def _load() -> None:
    if _REGISTRY:
        return
    from nqlab.strategies import baselines, bb_vp_macd  # noqa: F401

    for cls in (baselines.ORB, baselines.FirstHalfHourMomentum, baselines.RandomEntry,
                baselines.SessionHold, baselines.NullFixedTime,
                bb_vp_macd.BBVPMacdLimit, bb_vp_macd.BBVPMacdTrigger):
        register(cls)

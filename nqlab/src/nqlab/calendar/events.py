"""Scheduled macro release calendar (config/events.yaml).

Explicit date lists are used as given. Rule-based kinds (NFP = first Friday, ISM = first
business day) are approximations and every session they touch is flagged
``events_approximate`` so reports can say so.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from nqlab.settings import Settings, load_yaml


def _first_weekday(year: int, month: int, weekday: int) -> dt.date:
    d = dt.date(year, month, 1)
    while d.weekday() != weekday:
        d += dt.timedelta(days=1)
    return d


def _first_business_day(year: int, month: int) -> dt.date:
    d = dt.date(year, month, 1)
    while d.weekday() >= 5 or (d.month == 1 and d.day == 1):
        d += dt.timedelta(days=1)
    return d


def _dates(xs) -> set[dt.date]:
    return {x if isinstance(x, dt.date) else dt.date.fromisoformat(str(x)) for x in (xs or [])}


@dataclass
class EventCalendar:
    fomc: set[dt.date] = field(default_factory=set)
    r0830: dict[str, set[dt.date]] = field(default_factory=dict)
    r1000: dict[str, set[dt.date]] = field(default_factory=dict)
    rules: dict[str, str] = field(default_factory=dict)
    verified: bool = False

    @classmethod
    def load(cls, settings: Settings) -> "EventCalendar":
        p = settings.path("events")
        raw = load_yaml(p) if p.exists() else {}
        return cls(
            fomc=_dates(raw.get("fomc")),
            r0830={k: _dates(v) for k, v in (raw.get("release_0830") or {}).items()},
            r1000={k: _dates(v) for k, v in (raw.get("release_1000") or {}).items()},
            rules=dict(raw.get("rules") or {}),
            verified=bool(raw.get("verified", False)),
        )

    def _rule_hit(self, kind: str, d: dt.date) -> bool:
        rule = self.rules.get(kind)
        if rule == "first_friday":
            return d == _first_weekday(d.year, d.month, 4)
        if rule == "first_business_day":
            return d == _first_business_day(d.year, d.month)
        return False

    def flags(self, d: dt.date) -> dict:
        kinds: list[str] = []
        approx = False
        for group, table in (("0830", self.r0830), ("1000", self.r1000)):
            names = set(table) | {k for k in self.rules if (k in ("nfp",) and group == "0830")
                                  or (k in ("ism_manufacturing",) and group == "1000")}
            for kind in sorted(names):
                explicit = table.get(kind) or set()
                if explicit:
                    if d in explicit:
                        kinds.append(f"{kind}@{group}")
                elif self._rule_hit(kind, d):
                    kinds.append(f"{kind}@{group}")
                    approx = True
        return {
            "is_fomc": d in self.fomc,
            "is_release_0830": any(k.endswith("@0830") for k in kinds),
            "is_release_1000": any(k.endswith("@1000") for k in kinds),
            "release_kinds": ",".join(kinds),
            "events_approximate": approx or not self.verified,
        }

"""The trial ledger: append-only, hash-chained record of every trial ever run.

The Deflated Sharpe Ratio is only as honest as its N. N comes from here and nowhere
else. This module deliberately exposes no way to delete or edit rows; a test asserts no
DELETE/UPDATE/DROP/TRUNCATE statement touching ``trial_ledger`` exists in the codebase.
``verify()`` detects rows edited or removed outside this module (the hash chain breaks).

Trial counting
--------------
``trial_key`` identifies a distinct hypothesis: (family, strategy, params, origin
variant). The same config re-run under a different fill tier or data version is the
same trial. LLM variants are reserved at generation time with the trial keys their
grid would produce, so running them later does not double count and *not* running
them still counts.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from nqlab.hashing import canonical_json, stable_hash
from nqlab.results.db import connect

# Kinds that represent candidate strategies and therefore count toward DSR's N.
COUNTED_KINDS = ("config_run", "llm_variant", "holdout")
KINDS = COUNTED_KINDS + ("permutation", "diagnostic", "calibration", "report")
STATUSES = ("reserved", "completed", "errored", "abandoned")
ORIGINS = ("human", "llm_variant")

GENESIS = "0" * 64

_COLS = (
    "seq", "ts_utc", "trial_key", "run_id", "family_id", "strategy", "params_json",
    "kind", "status", "origin", "data_version", "fill_tier", "sharpe", "n_obs", "note",
    "prev_hash", "row_hash",
)


def trial_key(family_id: str | None, strategy: str, params: dict[str, Any], variant: str | None = None) -> str:
    return stable_hash({"f": family_id, "s": strategy, "p": params, "v": variant}, n=24)


@dataclass
class LedgerEntry:
    trial_key: str
    kind: str
    status: str
    origin: str = "human"
    run_id: str | None = None
    family_id: str | None = None
    strategy: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    data_version: str | None = None
    fill_tier: int | None = None
    sharpe: float | None = None
    n_obs: int | None = None
    note: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown ledger kind {self.kind!r}")
        if self.status not in STATUSES:
            raise ValueError(f"unknown ledger status {self.status!r}")
        if self.origin not in ORIGINS:
            raise ValueError(f"unknown ledger origin {self.origin!r}")


def _row_hash(prev_hash: str, payload: dict[str, Any]) -> str:
    return hashlib.sha256((prev_hash + canonical_json(payload)).encode("utf-8")).hexdigest()


class TrialLedger:
    """Append-only. There is intentionally no delete, update or truncate method."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)

    def append(self, entries: Iterable[LedgerEntry]) -> int:
        entries = list(entries)
        if not entries:
            return 0
        with connect(self.db_path) as con:
            con.execute("BEGIN TRANSACTION")
            try:
                last = con.execute(
                    "SELECT row_hash FROM trial_ledger ORDER BY seq DESC LIMIT 1"
                ).fetchone()
                prev = last[0] if last else GENESIS
                rows = []
                for e in entries:
                    seq = con.execute("SELECT nextval('ledger_seq')").fetchone()[0]
                    ts = dt.datetime.now(dt.timezone.utc)
                    payload = {
                        "seq": seq, "ts_utc": ts.isoformat(), "trial_key": e.trial_key,
                        "run_id": e.run_id, "family_id": e.family_id, "strategy": e.strategy,
                        "params_json": canonical_json(e.params), "kind": e.kind,
                        "status": e.status, "origin": e.origin, "data_version": e.data_version,
                        "fill_tier": e.fill_tier,
                        "sharpe": None if e.sharpe is None else float(e.sharpe),
                        "n_obs": e.n_obs, "note": e.note,
                    }
                    h = _row_hash(prev, payload)
                    rows.append((seq, ts, e.trial_key, e.run_id, e.family_id, e.strategy,
                                 payload["params_json"], e.kind, e.status, e.origin,
                                 e.data_version, e.fill_tier, payload["sharpe"], e.n_obs,
                                 e.note, prev, h))
                    prev = h
                con.executemany(
                    f"INSERT INTO trial_ledger VALUES ({','.join('?' * len(_COLS))})", rows
                )
                con.execute("COMMIT")
            except Exception:
                con.execute("ROLLBACK")
                raise
        return len(entries)

    # ---- reads -------------------------------------------------------------------
    def n_trials(self, family_id: str | None = None) -> int:
        """Distinct candidate trials. Global by default: the research program as a whole
        is what is being searched, so the honest N is everything ever tried."""
        q = (
            "SELECT COUNT(DISTINCT trial_key) FROM trial_ledger WHERE kind IN "
            f"({','.join('?' * len(COUNTED_KINDS))})"
        )
        args: list[Any] = list(COUNTED_KINDS)
        if family_id is not None:
            q += " AND family_id = ?"
            args.append(family_id)
        if not self.db_path.exists():
            return 0
        with connect(self.db_path) as con:
            return int(con.execute(q, args).fetchone()[0])

    def trial_sharpes(self) -> list[tuple[float, int]]:
        """(per-period Sharpe, n_obs) of every completed counted trial, latest per key."""
        if not self.db_path.exists():
            return []
        with connect(self.db_path) as con:
            rows = con.execute(
                f"""SELECT sharpe, n_obs FROM (
                      SELECT trial_key, sharpe, n_obs,
                             row_number() OVER (PARTITION BY trial_key ORDER BY seq DESC) rn
                      FROM trial_ledger
                      WHERE status='completed' AND sharpe IS NOT NULL
                        AND kind IN ({','.join('?' * len(COUNTED_KINDS))})
                    ) WHERE rn = 1""",
                list(COUNTED_KINDS),
            ).fetchall()
        return [(float(s), int(n or 0)) for s, n in rows]

    def stats(self) -> dict[str, Any]:
        if not self.db_path.exists():
            return {"n_trials": 0, "rows": 0, "by_kind": {}, "by_status": {}, "by_origin": {}}
        with connect(self.db_path) as con:
            rows = con.execute("SELECT COUNT(*) FROM trial_ledger").fetchone()[0]
            by_kind = dict(con.execute("SELECT kind, COUNT(*) FROM trial_ledger GROUP BY 1").fetchall())
            by_status = dict(con.execute("SELECT status, COUNT(*) FROM trial_ledger GROUP BY 1").fetchall())
            by_origin = dict(con.execute(
                "SELECT origin, COUNT(DISTINCT trial_key) FROM trial_ledger GROUP BY 1").fetchall())
        return {"n_trials": self.n_trials(), "rows": rows, "by_kind": by_kind,
                "by_status": by_status, "by_origin": by_origin}

    def stage_outcomes(self, family_id: str) -> list[tuple[int, str]]:
        """Stage outcomes recorded in the hash-chained ledger (source of truth for
        archiving: editing the families/stage_results tables cannot undo a FAIL)."""
        if not self.db_path.exists():
            return []
        with connect(self.db_path) as con:
            rows = con.execute("SELECT note FROM trial_ledger WHERE family_id=? AND trial_key LIKE 'stage:%' "
                               "ORDER BY seq", [family_id]).fetchall()
        out = []
        for (note,) in rows:
            try:
                head = note.split(":", 1)[0]          # "STAGE 3 FAIL"
                _, st, status = head.split(" ", 2)
                out.append((int(st), status))
            except Exception:
                continue
        return out

    def verify(self) -> tuple[bool, str]:
        """Recompute the hash chain. Any edit or deletion outside append() breaks it."""
        if not self.db_path.exists():
            return True, "empty"
        with connect(self.db_path) as con:
            rows = con.execute(f"SELECT {','.join(_COLS)} FROM trial_ledger ORDER BY seq").fetchall()
        prev = GENESIS
        for r in rows:
            rec = dict(zip(_COLS, r))
            if rec["prev_hash"] != prev:
                return False, f"chain broken before seq {rec['seq']}"
            payload = {
                "seq": rec["seq"], "ts_utc": rec["ts_utc"].astimezone(dt.timezone.utc).isoformat(),
                "trial_key": rec["trial_key"], "run_id": rec["run_id"],
                "family_id": rec["family_id"], "strategy": rec["strategy"],
                "params_json": rec["params_json"], "kind": rec["kind"], "status": rec["status"],
                "origin": rec["origin"], "data_version": rec["data_version"],
                "fill_tier": rec["fill_tier"], "sharpe": rec["sharpe"], "n_obs": rec["n_obs"],
                "note": rec["note"],
            }
            if _row_hash(prev, payload) != rec["row_hash"]:
                return False, f"row {rec['seq']} was modified"
            prev = rec["row_hash"]
        return True, f"{len(rows)} rows verified"

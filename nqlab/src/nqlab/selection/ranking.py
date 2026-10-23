"""Rank families by P(reaching first payout), never by Sharpe (spec 9). Every other
headline number is shown next to it so no single metric is read alone."""

from __future__ import annotations

from typing import Any

from nqlab.results.db import connect
from nqlab.selection.family import FamilyStore


def rank_families(settings) -> list[dict[str, Any]]:
    store = FamilyStore(settings)
    rows = []
    with connect(settings.path("results_db")) as con:
        for f in store.list():
            fid = f["family_id"]
            res = {r["stage"]: r for r in store.results(fid)}
            s5 = res.get(5, {}).get("details", {})
            s6 = res.get(6, {}).get("details", {})
            run = con.execute("SELECT run_id FROM runs WHERE family_id=? AND fill_tier=2 ORDER BY ts_utc DESC LIMIT 1",
                              [fid]).fetchone()
            m = {}
            if run:
                m = {k: v for k, v, _t in con.execute("SELECT key, value, text FROM metrics WHERE run_id=?", [run[0]]).fetchall()}
            last = max(res) if res else 0
            rows.append({
                "family_id": fid, "strategy": f["strategy"], "status": f["status"],
                "furthest_stage": last, "last_result": res[last]["status"] if res else "not run",
                "p_first_payout": s6.get("p_first_payout"), "zero_edge_p_first_payout": s6.get("zero_edge_p_first_payout"),
                "p_pass": (s6.get("bootstrap") or {}).get("p_pass"),
                "sharpe_ann": m.get("sharpe_ann"), "sharpe_ann_se": m.get("sharpe_ann_se"),
                "dsr": (s5.get("dsr") or {}).get("dsr"), "expectancy_r": m.get("expectancy_r"),
                "expectancy_usd": m.get("expectancy_usd"), "profit_factor": m.get("profit_factor"),
                "max_consec_losses": m.get("max_consec_losses"), "n_trades": m.get("n_trades"),
            })
    rows.sort(key=lambda r: (r["p_first_payout"] is None, -(r["p_first_payout"] or 0)))
    return rows

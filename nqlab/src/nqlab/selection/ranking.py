"""Rank families by expected net dollars per purchased account after all fees, judged with
the edge haircut to its one-SE lower bound (the farming objective); never by Sharpe.
P(first payout) and every other headline number are shown beside it."""

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
                "ev_net_per_account": s6.get("ev_net_per_account"),
                "ev_net_per_account_haircut": s6.get("ev_net_per_account_haircut"),
                "zero_edge_ev_net": s6.get("zero_edge_ev_net_per_account"),
                "payout_path": s6.get("farming_best_path"),
                "p_first_payout": s6.get("p_first_payout"), "zero_edge_p_first_payout": s6.get("zero_edge_p_first_payout"),
                "p_pass": (s6.get("bootstrap") or {}).get("p_pass"),
                "sharpe_ann": m.get("sharpe_ann"), "sharpe_ann_se": m.get("sharpe_ann_se"),
                "dsr": (s5.get("dsr") or {}).get("dsr"), "expectancy_r": m.get("expectancy_r"),
                "expectancy_usd": m.get("expectancy_usd"), "profit_factor": m.get("profit_factor"),
                "max_consec_losses": m.get("max_consec_losses"), "n_trades": m.get("n_trades"),
            })
    rows.sort(key=lambda r: (r["ev_net_per_account_haircut"] is None, -(r["ev_net_per_account_haircut"] or 0),
                             -(r["p_first_payout"] or 0)))
    return rows

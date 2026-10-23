"""Numba event loop over 1-minute bars. One position at a time, flat at session end.

Order of events within bar i
----------------------------
1. Session change: cancel pendings (a position cannot survive a session boundary; the
   previous session's last bar closed it).
2. Open position: forced flatten / exit signal fill at the OPEN of bar i (+slippage);
   otherwise stop/target are checked against bar i's range.
3. Flat: pending entry orders (placed at the close of an EARLIER bar) are checked
   against bar i. On a fill, exits are checked on the remainder of bar i using only
   what the bar's continuity can prove (see below).
4. Expiry of pending orders.
5. Signals computed at the close of bar i become pending orders for bar i+1 onward.
   This ordering is what guarantees invariant 1: no same-bar fills.

Fill rules (tier 1 = touch, tier 2 = trade-through, `through` = 0 or 1 tick)
--------------------------------------------------------------------------
* Limit buy at P fills when low <= P - through. If open <= P - through it filled at the
  open (gap), otherwise at P. Sells mirror.
* Stop buy at P triggers when high >= P; fills at max(open, P) + slippage.
* Market fills at the next bar's open + slippage.
* Stop-limit: triggers like a stop; fills at max(open, P) if that is <= limit, else
  rests as a limit at the limit price from the next bar.
* Exits: stop fills at the stop (or the open, if gapped through) minus slippage; target
  fills like a limit (trade-through for tier 2).
* Ambiguous bar: if a bar touches both stop and target, the STOP filled first. Counted.
* Entry bar: after a limit fill intrabar, the bar's high may have printed BEFORE the
  fill, so the target is not allowed to fill on the entry bar; the stop is (price had to
  pass P to reach a stop beyond it). After a stop-entry intrabar both are allowed (the
  target lies beyond the trigger) and both-touched resolves to the stop. After a fill at
  the open, the whole bar is post-fill.
* MAE includes the full range of every bar the position was open, except that a stop
  exit caps it at the exit price. MFE on a limit-entry bar uses the close, not the high.
"""

from __future__ import annotations

import numba as nb
import numpy as np

NONE, MARKET, LIMIT, STOP, STOPLIMIT = 0, 1, 2, 3, 4
R_STOP, R_TARGET, R_FLATTEN, R_EXIT, R_DATA_END = 1, 2, 3, 4, 5
K_OPEN, K_LIMIT, K_STOP = 0, 1, 2   # how the entry filled (for entry-bar rules)


@nb.njit(cache=True)
def _fill_entry(side, typ, px, lim, o, h, l, through, tick, slip):
    """Return (filled, fill_price, kind, new_type_if_converted). side=+1 buy, -1 sell."""
    if typ == MARKET:
        return True, o + side * slip * tick, K_OPEN, typ
    if typ == LIMIT:
        thr = px - side * through
        if side > 0:
            if o <= thr:
                return True, o, K_OPEN, typ
            if l <= thr:
                return True, px, K_LIMIT, typ
        else:
            if o >= thr:
                return True, o, K_OPEN, typ
            if h >= thr:
                return True, px, K_LIMIT, typ
        return False, 0.0, 0, typ
    if typ == STOP or typ == STOPLIMIT:
        trig = (h >= px) if side > 0 else (l <= px)
        if not trig:
            return False, 0.0, 0, typ
        gap = (o >= px) if side > 0 else (o <= px)
        base = o if gap else px
        kind = K_OPEN if gap else K_STOP
        if typ == STOP:
            return True, base + side * slip * tick, kind, typ
        if (side > 0 and base <= lim) or (side < 0 and base >= lim):
            return True, base, kind, typ
        return False, 0.0, 0, LIMIT   # converts to a resting limit at `lim`
    return False, 0.0, 0, typ


@nb.njit(cache=True)
def run_kernel(o, h, l, c, sess, slip, force_flat, entry_ok,
               l_type, l_px, l_lim, l_stop, l_tgt,
               s_type, s_px, s_lim, s_stop, s_tgt,
               cancel_l, cancel_s, exit_now,
               tier, tick, expiry_bars, max_trades, stop_after_win, stop_after_loss,
               be_trigger_r, be_offset_ticks):
    n = o.shape[0]
    through = tick if tier >= 2 else 0.0
    cap = n // 2 + 2
    t_entry = np.empty(cap, np.int64)
    t_exit = np.empty(cap, np.int64)
    t_side = np.empty(cap, np.int64)
    t_epx = np.empty(cap, np.float64)
    t_xpx = np.empty(cap, np.float64)
    t_sd = np.empty(cap, np.float64)
    t_td = np.empty(cap, np.float64)
    t_mae = np.empty(cap, np.float64)
    t_mfe = np.empty(cap, np.float64)
    t_reason = np.empty(cap, np.int64)
    t_kind = np.empty(cap, np.int64)
    t_amb = np.empty(cap, np.bool_)
    t_be = np.empty(cap, np.bool_)
    nt = 0

    # pending orders: index 0 long, 1 short
    p_type = np.zeros(2, np.int64)
    p_px = np.zeros(2)
    p_lim = np.zeros(2)
    p_sd = np.zeros(2)
    p_td = np.zeros(2)
    p_at = np.zeros(2, np.int64)

    pos = 0
    e_px = 0.0
    stop_px = 0.0
    tgt_px = np.nan
    sd = 0.0
    td = np.nan
    e_idx = 0
    mae = 0.0
    mfe = 0.0
    amb = False
    be_done = False
    exit_flag = False
    trades_today = 0
    done_today = False

    for i in range(n):
        new_sess = i == 0 or sess[i] != sess[i - 1]
        last_of_sess = i == n - 1 or sess[i + 1] != sess[i]
        if new_sess:
            p_type[0] = NONE
            p_type[1] = NONE
            trades_today = 0
            done_today = False
            exit_flag = False

        # ---------------- 2. manage open position ---------------------------------
        if pos != 0:
            exited = False
            xpx = 0.0
            reason = 0
            if force_flat[i] or exit_flag:
                xpx = o[i] - pos * slip[i] * tick
                reason = R_FLATTEN if force_flat[i] else R_EXIT
                exited = True
                # adverse excursion at the open only
                adv = (e_px - o[i]) * pos
                if adv > mae:
                    mae = adv
            else:
                res = _exit_check(pos, stop_px, tgt_px, o[i], h[i], l[i], through, tick, slip[i], True, True)
                exited, xpx, reason, a = res
                if a:
                    amb = True
                if exited and reason == R_STOP:
                    adv = (e_px - xpx) * pos
                    if adv > mae:
                        mae = adv
                else:
                    adv = (e_px - (l[i] if pos > 0 else h[i])) * pos
                    if adv > mae:
                        mae = adv
                fav = ((h[i] if pos > 0 else l[i]) - e_px) * pos
                if fav > mfe:
                    mfe = fav
            if exited:
                t_entry[nt] = e_idx; t_exit[nt] = i; t_side[nt] = pos; t_epx[nt] = e_px
                t_xpx[nt] = xpx; t_sd[nt] = sd; t_td[nt] = td; t_mae[nt] = mae; t_mfe[nt] = mfe
                t_reason[nt] = reason; t_amb[nt] = amb; t_be[nt] = be_done
                pnl = (xpx - e_px) * pos
                nt += 1
                pos = 0
                exit_flag = False
                if (stop_after_win and pnl > 0) or (stop_after_loss and pnl < 0):
                    done_today = True

        # ---------------- 3. pending entries ----------------------------------------
        if pos == 0 and not force_flat[i]:
            f0, px0, k0, nt0 = (False, 0.0, 0, p_type[0])
            f1, px1, k1, nt1 = (False, 0.0, 0, p_type[1])
            if p_type[0] != NONE:
                f0, px0, k0, nt0 = _fill_entry(1, p_type[0], p_px[0], p_lim[0], o[i], h[i], l[i], through, tick, slip[i])
            if p_type[1] != NONE:
                f1, px1, k1, nt1 = _fill_entry(-1, p_type[1], p_px[1], p_lim[1], o[i], h[i], l[i], through, tick, slip[i])
            both = f0 and f1
            pick = -1
            if both:
                # ambiguous: take the side whose trigger is nearer the open; the stop
                # check below then usually stops it out on this bar (pessimistic)
                pick = 0 if abs(px0 - o[i]) <= abs(px1 - o[i]) else 1
            elif f0:
                pick = 0
            elif f1:
                pick = 1
            if not f0 and nt0 != p_type[0]:
                p_type[0] = nt0; p_px[0] = p_lim[0]; p_at[0] = i
            if not f1 and nt1 != p_type[1]:
                p_type[1] = nt1; p_px[1] = p_lim[1]; p_at[1] = i
            if pick >= 0:
                side = 1 if pick == 0 else -1
                e_px = px0 if pick == 0 else px1
                kind = k0 if pick == 0 else k1
                sd = p_sd[pick]
                td = p_td[pick]
                stop_px = e_px - side * sd if sd == sd else np.nan
                tgt_px = e_px + side * td if td == td else np.nan
                pos = side
                e_idx = i
                amb = both
                be_done = False
                mae = 0.0
                mfe = 0.0
                trades_today += 1
                t_kind[nt] = kind
                # all pendings cancelled on fill (one position at a time; OCO implicit)
                p_type[0] = NONE
                p_type[1] = NONE
                allow_tgt = kind != K_LIMIT
                res = _exit_check(pos, stop_px, tgt_px, o[i] if kind == K_OPEN else e_px,
                                  h[i], l[i], through, tick, slip[i], True, allow_tgt)
                exited, xpx, reason, a = res
                if a:
                    amb = True
                adv = (e_px - (l[i] if pos > 0 else h[i])) * pos
                if exited and reason == R_STOP:
                    adv = (e_px - xpx) * pos
                if adv > mae:
                    mae = adv
                if kind == K_LIMIT:
                    fav = (c[i] - e_px) * pos
                else:
                    fav = ((h[i] if pos > 0 else l[i]) - e_px) * pos
                if exited and reason == R_TARGET:
                    fav = max(fav, (xpx - e_px) * pos)
                if fav > mfe:
                    mfe = fav
                if exited:
                    t_entry[nt] = e_idx; t_exit[nt] = i; t_side[nt] = pos; t_epx[nt] = e_px
                    t_xpx[nt] = xpx; t_sd[nt] = sd; t_td[nt] = td; t_mae[nt] = mae; t_mfe[nt] = mfe
                    t_reason[nt] = reason; t_amb[nt] = amb; t_be[nt] = False
                    pnl = (xpx - e_px) * pos
                    nt += 1
                    pos = 0
                    if (stop_after_win and pnl > 0) or (stop_after_loss and pnl < 0):
                        done_today = True
        elif force_flat[i]:
            p_type[0] = NONE
            p_type[1] = NONE

        # ---------------- 4. expiry -------------------------------------------------------
        if expiry_bars > 0:
            for k in range(2):
                if p_type[k] != NONE and i - p_at[k] >= expiry_bars:
                    p_type[k] = NONE

        # ---------------- 6. session end without a flatten bar (missing data) -------------
        if last_of_sess:
            if pos != 0:
                xpx = c[i] - pos * slip[i] * tick
                adv = (e_px - xpx) * pos
                if adv > mae:
                    mae = adv
                t_entry[nt] = e_idx; t_exit[nt] = i; t_side[nt] = pos; t_epx[nt] = e_px
                t_xpx[nt] = xpx; t_sd[nt] = sd; t_td[nt] = td; t_mae[nt] = mae; t_mfe[nt] = mfe
                t_reason[nt] = R_DATA_END; t_amb[nt] = amb; t_be[nt] = be_done
                nt += 1
                pos = 0
            p_type[0] = NONE
            p_type[1] = NONE
            continue

        # ---------------- 5. signals at the close of bar i ------------------------------
        if force_flat[i]:
            continue
        if pos != 0:
            if exit_now[i]:
                exit_flag = True
            if be_trigger_r == be_trigger_r and not be_done and sd == sd and mfe >= be_trigger_r * sd:
                stop_px = e_px + pos * be_offset_ticks * tick
                be_done = True
            continue
        if cancel_l[i]:
            p_type[0] = NONE
        if cancel_s[i]:
            p_type[1] = NONE
        if done_today or trades_today >= max_trades or not entry_ok[i]:
            continue
        if l_type[i] != NONE:
            p_type[0] = l_type[i]; p_px[0] = l_px[i]; p_lim[0] = l_lim[i]
            p_sd[0] = l_stop[i]; p_td[0] = l_tgt[i]; p_at[0] = i
        if s_type[i] != NONE:
            p_type[1] = s_type[i]; p_px[1] = s_px[i]; p_lim[1] = s_lim[i]
            p_sd[1] = s_stop[i]; p_td[1] = s_tgt[i]; p_at[1] = i

    return (t_entry[:nt], t_exit[:nt], t_side[:nt], t_epx[:nt], t_xpx[:nt], t_sd[:nt], t_td[:nt],
            t_mae[:nt], t_mfe[:nt], t_reason[:nt], t_kind[:nt], t_amb[:nt], t_be[:nt])


@nb.njit(cache=True)
def _exit_check(pos, stop_px, tgt_px, o, h, l, through, tick, slip, allow_stop, allow_tgt):
    """Returns (exited, price, reason, ambiguous). ``o`` is the reference open (for an
    intrabar entry it is the entry price, so no gap logic applies)."""
    has_stop = allow_stop and stop_px == stop_px
    has_tgt = allow_tgt and tgt_px == tgt_px
    if pos > 0:
        if has_stop and o <= stop_px:
            return True, o - slip * tick, R_STOP, False
        if has_tgt and o >= tgt_px + through:
            return True, o, R_TARGET, False
        s_hit = has_stop and l <= stop_px
        t_hit = has_tgt and h >= tgt_px + through
    else:
        if has_stop and o >= stop_px:
            return True, o + slip * tick, R_STOP, False
        if has_tgt and o <= tgt_px - through:
            return True, o, R_TARGET, False
        s_hit = has_stop and h >= stop_px
        t_hit = has_tgt and l <= tgt_px - through
    if s_hit and t_hit:
        return True, stop_px - pos * slip * tick, R_STOP, True
    if s_hit:
        return True, stop_px - pos * slip * tick, R_STOP, False
    if t_hit:
        return True, tgt_px, R_TARGET, False
    return False, 0.0, 0, False

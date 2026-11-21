"""Unabhaengige Python-Gegenrechnung des Strategie-Testers (strategie.js), Regeln: SPEC_STATISTIK.md Abschnitt 5.

Eigener Code mit pandas (Indikatoren ueber rolling statt Schleifen). Wird nur in den Tests benutzt:
tests/test_strategie.py vergleicht diese Ergebnisse mit strategie.js (Node) auf denselben Daten.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _cmp(v: pd.Series, op: str, th: float) -> pd.Series:
    out = (v > th) if op == "gt" else (v < th)
    return out & v.notna() & np.isfinite(v.astype(float))


def rsi_wilder(x: pd.Series, n: int) -> pd.Series:
    ch = x.diff()
    gain, loss = ch.clip(lower=0), (-ch).clip(lower=0)
    out = pd.Series(np.nan, index=x.index)
    if len(x) <= n:
        return out
    g, lo = gain.iloc[1:n + 1].mean(), loss.iloc[1:n + 1].mean()

    def val(g, lo):
        return (50.0 if g == 0 else 100.0) if lo == 0 else 100 - 100 / (1 + g / lo)
    out.iloc[n] = val(g, lo)
    for i in range(n + 1, len(x)):
        g = (g * (n - 1) + gain.iloc[i]) / n
        lo = (lo * (n - 1) + loss.iloc[i]) / n
        out.iloc[i] = val(g, lo)
    return out


def condition(cond: dict, x: pd.Series, interval: str, crypto: bool) -> pd.Series:
    """x: Schlusskurse an den Periodenenden (Index = Datum). Ergebnis: bool je Periodenende."""
    n = max(1, round(cond.get("n", 1)))
    k = {"1d": 365 if crypto else 252, "1wk": 52, "1mo": 12}[interval]
    r = x / x.shift(1) - 1
    typ = cond["type"]
    if typ == "always":
        return pd.Series(True, index=x.index)
    if typ == "price_sma":
        return _cmp(x - x.rolling(n).mean(), cond["op"], 0)
    if typ == "sma_sma":
        a, b = max(1, round(cond["a"])), max(1, round(cond["b"]))
        return _cmp(x.rolling(a).mean() - x.rolling(b).mean(), cond["op"], 0)
    if typ == "mom":
        return _cmp(x / x.shift(n) - 1, cond["op"], cond["th"])
    if typ == "ret":
        return _cmp(r, cond["op"], cond["th"])
    if typ == "z":
        if n < 2:
            return pd.Series(False, index=x.index)
        prev = r.shift(1)
        mu, sd = prev.rolling(n).mean(), prev.rolling(n).std(ddof=1)
        z = (r - mu) / sd.where(sd > 0)
        return _cmp(z, cond["op"], cond["th"])
    if typ == "rsi":
        return _cmp(rsi_wilder(x, n), cond["op"], cond["th"])
    if typ == "hilo":
        if cond.get("side") == "low":
            v = x / x.rolling(n).min() - 1
            return (v <= cond["th"]) & v.notna()
        v = x / x.rolling(n).max() - 1
        return (v >= -cond["th"]) & v.notna()
    if typ == "vol":
        if n < 2:
            return pd.Series(False, index=x.index)
        return _cmp(r.rolling(n).std(ddof=1) * math.sqrt(k), cond["op"], cond["th"])
    if typ == "calendar":
        ok = pd.Series(True, index=x.index)
        if cond.get("months"):
            ok &= x.index.month.isin(cond["months"])
        if interval == "1d" and cond.get("weekdays"):
            ok &= (x.index.weekday + 1).isin(cond["weekdays"])
        return ok
    raise ValueError(typ)


def backtest(dates: pd.DatetimeIndex, close: np.ndarray, ends: dict, cfg: dict, crypto: bool = False) -> dict:
    c = np.asarray(close, dtype=float)
    T = c.size
    lo = int(np.searchsorted(dates.values, np.datetime64(cfg["start"]), "left")) if cfg.get("start") else 0
    hi = int(np.searchsorted(dates.values, np.datetime64(cfg["end"]), "right")) - 1 if cfg.get("end") else T - 1
    iv = cfg["interval"]
    E = list(range(T)) if iv == "1d" else list(ends[iv])
    x = pd.Series(c[E], index=dates[E])
    conds = cfg.get("conds") or [{"type": "always"}]
    sig = np.ones(len(E), dtype=bool)
    for cd in conds:
        sig &= condition(cd, x, iv, crypto).to_numpy(dtype=bool)
    eval_j = {i: j for j, i in enumerate(E)}
    size = cfg.get("size", 1) * cfg.get("lev", 1)
    k = cfg.get("cost", {})
    side = (k.get("fee_bp", 0) + k.get("spread_bp", 0) / 2 + k.get("slip_bp", 0)) / 1e4 * size
    fin = k.get("fin_pa", 0) / 100 * size
    ex = cfg.get("exit", {})
    lag = cfg.get("lag", 1)
    d = cfg.get("dir", "long")

    def want(j):
        if d == "ls":
            return 1 if sig[j] else -1
        return (-1 if d == "short" else 1) if sig[j] else 0

    cal = np.diff(dates.values).astype("timedelta64[D]").astype(float)
    g = np.zeros(T)
    held = np.zeros(T, dtype=int)
    st = {"pos": 0, "pend": None, "px": 0.0, "peak": 0.0, "ej": -1, "trade": None}
    trades = []

    def execute(target, i, reason):
        dp = abs(target - st["pos"])
        if dp == 0:
            return
        g[i] -= dp * side
        if st["pos"] != 0 and st["trade"] is not None:
            tr = st["trade"]
            tr.update(exit=i, reason=reason, growth=tr["growth"] * (1 - side))
            trades.append(tr)
            st["trade"] = None
        st["pos"] = target
        if target != 0:
            st["px"] = st["peak"] = c[i]
            st["trade"] = {"dir": target, "entry": i, "growth": 1 - side}

    def order(target, i, reason):
        if lag == 0:
            execute(target, i, reason)
        elif i + lag <= hi:
            st["pend"] = (target, i + lag, reason)

    eq, ruined = 1.0, False
    for i in range(lo, hi + 1):
        out_today = False
        if i > lo and not ruined:
            a = size * st["pos"] * (c[i] / c[i - 1] - 1) - fin * cal[i - 1] / 365 if st["pos"] else 0.0
            g[i] += a
            held[i] = st["pos"]
            if st["trade"] is not None:
                st["trade"]["growth"] *= 1 + a
        if not ruined:
            if st["pend"] is not None and st["pend"][1] == i:
                tgt, _, why = st["pend"]
                before = st["pos"]
                execute(tgt, i, why)
                out_today = before != 0 and tgt != before
                st["pend"] = None
            tr = st["trade"]
            if st["pos"] and st["pend"] is None and tr is not None and tr["entry"] < i:
                p = st["pos"]
                ret = p * (c[i] / st["px"] - 1)
                st["peak"] = max(st["peak"], c[i]) if p > 0 else min(st["peak"], c[i])
                why = None
                if ex.get("sl", 0) > 0 and ret <= -ex["sl"] / 100:
                    why = "Stop"
                elif ex.get("tp", 0) > 0 and ret >= ex["tp"] / 100:
                    why = "Ziel"
                elif ex.get("trail", 0) > 0 and p * (c[i] / st["peak"] - 1) <= -ex["trail"] / 100:
                    why = "Trailing"
                if why:
                    out_today = True
                    order(0, i, why)
            j = eval_j.get(i)
            if j is not None and st["pend"] is None:
                w = want(j)
                if st["pos"]:
                    if d == "ls" and w != st["pos"]:
                        if i + lag <= hi:
                            st["ej"] = j
                            order(w, i, "Wechsel")
                    elif ex.get("signal") and w != st["pos"]:
                        out_today = True
                        order(0, i, "Signal")
                    elif ex.get("maxHold", 0) > 0 and j - st["ej"] >= ex["maxHold"]:
                        out_today = True
                        order(0, i, "Haltedauer")
                elif w != 0 and not out_today and lo < i + lag <= hi:
                    st["ej"] = j
                    order(w, i, "Einstieg")
        eq *= 1 + g[i]
        if not ruined and eq <= 0:
            ruined = True
            eq = 0.0
            g[i] = -1.0  # Verlust am Ruin-Tag auf genau -100 % begrenzen
            st["pos"], st["pend"] = 0, None
            if st["trade"] is not None:
                st["trade"].update(exit=i, reason="Ruin", growth=0.0)
                trades.append(st["trade"])
                st["trade"] = None
    if st["trade"] is not None:
        st["trade"].update(exit=hi, reason="Ende")
        trades.append(st["trade"])

    days = np.arange(lo + 1, hi + 1)
    gs = g[days]
    kd = 365 if crypto else 252
    t0 = dates[lo]
    split = pd.Timestamp(cfg["split"]) if cfg.get("split") else t0 + 0.7 * (dates[hi] - t0)
    m_is = dates[days] < split
    res = {"g": gs, "days": days, "all": metrics(gs, dates[days], t0, kd),
           "bh": metrics(c[days] / c[days - 1] - 1, dates[days], t0, kd),
           "trades": [{"dir": t["dir"], "entryIdx": t["entry"], "exitIdx": t["exit"], "ret": t["growth"] - 1,
                       "reason": t["reason"]} for t in trades],
           "exposure": float((held[days] != 0).mean()), "lo": lo, "hi": hi}
    if m_is.sum() > 1:
        res["is"] = metrics(gs[m_is], dates[days][m_is], t0, kd)
    if (~m_is).sum() > 1:
        first = days[~m_is][0]
        res["oos"] = metrics(gs[~m_is], dates[days][~m_is], dates[first - 1], kd)
    return res


def metrics(g: np.ndarray, tt: pd.DatetimeIndex, t0: pd.Timestamp, k: int) -> dict:
    g = np.asarray(g, dtype=float)
    n = g.size
    eq = np.cumprod(1 + g)
    peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
    mdd = float(min(0.0, (eq / peak - 1).min()))
    # laengste Unterwasser-Phase in Kalendertagen (vom letzten Hoch bis zum Beobachtungstag)
    longest, last_peak_t, top = 0.0, t0, 1.0
    for e, t in zip(eq, tt):
        if e >= top:
            top, last_peak_t = e, t
        else:
            longest = max(longest, (t - last_peak_t).days)
    mean = float(g.mean())
    sd = float(g.std(ddof=1)) if n > 1 else None
    years = (tt[-1] - t0).days / 365.25
    e = float(eq[-1])
    cagr = e ** (1 / years) - 1 if e > 0 and years > 0 else -1.0
    dd = math.sqrt(float((np.minimum(g, 0) ** 2).mean()))
    from scipy import stats as st
    tstat = mean / (sd / math.sqrt(n)) if sd else None
    return {"n": n, "total": e - 1, "cagr": cagr, "vol": sd * math.sqrt(k) if sd is not None else None,
            "sharpe": mean / sd * math.sqrt(k) if sd else None, "sortino": mean / dd * math.sqrt(k) if dd > 0 else None,
            "maxdd": mdd, "dd_days": round(longest), "calmar": cagr / -mdd if mdd < 0 else None, "mean": mean,
            "t": tstat, "p": float(2 * st.t.sf(abs(tstat), n - 1)) if tstat is not None else None, "years": years}

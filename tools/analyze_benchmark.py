# -*- coding: utf-8 -*-
"""
██████╗  ██████╗ ██████╗ ████████╗██████╗        █████╗ ██╗
██╔══██╗██╔═══██╗██╔══██╗╚══██╔══╝██╔══██╗      ██╔══██╗██║
██████╔╝██║   ██║██████╔╝   ██║   ██████╔╝█████╗███████║██║
██╔═══╝ ██║   ██║██╔══██╗   ██║   ██╔══██╗╚════╝██╔══██║██║
██║     ╚██████╔╝██║  ██║   ██║   ██║  ██║      ██║  ██║██║
╚═╝      ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝      ╚═╝  ╚═╝╚═╝

auteur : @Claude (LLM)

File name : analyze_benchmark.py


Analyse approfondie d'un benchmark d'entraînement (CSV -> traitements -> figures + rapport).

Entrée
    training_benchmark.csv avec les colonnes : epoch, batch, step, loss, learning_rate,
    grad_norm, elapsed_seconds, tokens_per_second. Optionnelles : timestamp, event
    ("train" / "validation") et val_loss.

Traitements
    * Nettoyage : valeurs non finies, redémarrages (step qui recule), lignes rejouées
      après une reprise de checkpoint, vérification de cohérence de la reprise.
    * Loss : EMA, tendance robuste (polynôme local), bruit local, pics / chutes (z robuste),
      pente et plateau, loi de puissance avec perte irréductible + backtest + extrapolation,
      jalons de progression, perplexité.
    * Gradient : tendance, pics, corrélation avec la loss, taux de clipping optionnel.
    * Learning rate : durée de warmup, forme de la décroissance (cosinus / linéaire),
      fin de schedule estimée.
    * Débit : régime stable, dérive, surcoûts périodiques (checkpoint / éval), budget temps,
      temps perdu par un crash, ETA.
    * Statistiques : normalité des résidus, autocorrélation, périodogramme,
      corrélations, statistiques par segment.

Sorties (dans --out-dir)
    benchmark_overview.png, benchmark_loss.png, benchmark_optimization.png,
    benchmark_system.png, benchmark_stats.png, benchmark_processed.csv,
    benchmark_segments.csv, benchmark_anomalies.csv, benchmark_report.md,
    benchmark_report.json

Usage
    python analyze_benchmark.py [chemin/training_benchmark.csv] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT_DIR / "artifacts" / "benchmarks" / "training_benchmark.csv"

REQUIRED_COLUMNS = [
    "epoch",
    "batch",
    "step",
    "loss",
    "learning_rate",
    "grad_norm",
    "elapsed_seconds",
    "tokens_per_second",
]
TRAIN_NEEDED = ["step", "loss", "learning_rate", "grad_norm", "elapsed_seconds", "tokens_per_second"]

# Palette catégorielle validée (ordre fixe) + encres neutres.
C = {
    "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "magenta": "#e87ba4",
    "violet": "#6250d6", "red": "#e34948", "green": "#008300", "yellow": "#eda100",
    "blue_dark": "#184f95", "blue_soft": "#86b6ef",
    "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
    "grid": "#e1e0d9", "axis": "#c3c2b7",
}
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#2a78d6", "#0d366b"])
DIV_CMAP = LinearSegmentedColormap.from_list("div_bluered", ["#2a78d6", "#f0efec", "#e34948"])


# --------------------------------------------------------------------------- utilitaires
def robust_sigma(values) -> float:
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def lag1_autocorr(values) -> float:
    v = np.asarray(values, float)
    v = v - v.mean()
    denom = float((v * v).sum())
    if len(v) < 3 or denom <= 0:
        return 0.0
    return float((v[:-1] * v[1:]).sum() / denom)


def ols(x, y) -> dict:
    """Régression linéaire ; l'erreur-type est corrigée de l'autocorrélation (AR(1))."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    if n < 4 or np.ptp(x) == 0:
        return {"slope": float("nan"), "intercept": float("nan"), "r2": float("nan"), "se": float("nan"), "n": n}
    xm, ym = x.mean(), y.mean()
    sxx = float(((x - xm) ** 2).sum())
    slope = float(((x - xm) * (y - ym)).sum() / sxx)
    intercept = float(ym - slope * xm)
    res = y - (intercept + slope * x)
    sse = float((res**2).sum())
    sst = float(((y - ym) ** 2).sum())
    se = math.sqrt(sse / (n - 2) / sxx)
    r1 = min(max(lag1_autocorr(res), 0.0), 0.95)
    se *= math.sqrt((1 + r1) / (1 - r1))
    return {"slope": slope, "intercept": intercept, "r2": 1 - sse / sst if sst > 0 else float("nan"), "se": se, "n": n}


def local_poly(x, y, window: int, order: int = 2):
    """Lissage polynomial local (type Savitzky-Golay, abscisses quelconques).

    Renvoie (valeur lissée, dérivée première) ; aux bords la fenêtre est décalée."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(y)
    w = min(window, n if n % 2 else n - 1)
    w = max(w, order + 2)
    if w % 2 == 0:
        w += 1
    w = min(w, n)
    half = w // 2
    fit = np.empty(n)
    d1 = np.empty(n)
    for i in range(n):
        lo = min(max(0, i - half), n - w)
        hi = lo + w
        xs = x[lo:hi] - x[i]
        scale = float(np.abs(xs).max()) or 1.0
        coef = np.polyfit(xs / scale, y[lo:hi], order)
        fit[i] = coef[-1]
        d1[i] = coef[-2] / scale if order >= 1 else 0.0
    return fit, d1


def robust_trend(x, y, window: int, order: int = 2):
    """Tendance en deux passes : les points aberrants de la 1re passe sont neutralisés."""
    fit, d1 = local_poly(x, y, window, order)
    res = y - fit
    sigma = robust_sigma(res)
    if sigma > 0:
        mask = np.abs(res) > 3.5 * sigma
        if mask.any():
            fit, d1 = local_poly(x, np.where(mask, fit, y), window, order)
    return fit, d1


def ema(values, span: int):
    return pd.Series(values).ewm(span=max(span, 1), adjust=True).mean().to_numpy()


def fit_power_law(step, loss) -> dict:
    """loss = L_inf + A * step^-b  (L_inf par balayage, A et b par régression log-log)."""
    step = np.asarray(step, float)
    loss = np.asarray(loss, float)
    log_step = np.log(step)
    lmin = float(loss.min())

    def evaluate(l_inf):
        y = np.log(loss - l_inf)
        slope, intercept = np.polyfit(log_step, y, 1)
        pred = l_inf + np.exp(intercept + slope * log_step)
        return float(((loss - pred) ** 2).sum()), intercept, slope

    grid = np.linspace(0.0, lmin - 1e-3, 400)
    scores = np.array([evaluate(g)[0] for g in grid])
    best = int(np.argmin(scores))
    fine = np.linspace(grid[max(best - 1, 0)], grid[min(best + 1, len(grid) - 1)], 200)
    l_inf = float(fine[int(np.argmin([evaluate(g)[0] for g in fine]))])
    sse, intercept, slope = evaluate(l_inf)
    sst = float(((loss - loss.mean()) ** 2).sum())
    return {
        "l_inf": l_inf, "A": float(np.exp(intercept)), "b": float(-slope),
        "r2": 1 - sse / sst if sst > 0 else float("nan"),
        "rmse": math.sqrt(sse / len(loss)),
        "at_bound": bool(best in (0, len(grid) - 1)),
    }


def power_law(step, p: dict):
    return p["l_inf"] + p["A"] * np.asarray(step, float) ** (-p["b"])


def fit_cosine_decay(t, y, peak):
    """y = a + b*cos(k*t) (t relatif au pic). Balayage de k + moindres carrés pour (a, b)."""
    span = float(t.max())
    best = (float("inf"), None)

    def solve(ks):
        nonlocal best
        for k in ks:
            design = np.column_stack([np.ones_like(t), np.cos(k * t)])
            coef, *_ = np.linalg.lstsq(design, y, rcond=None)
            sse = float(((design @ coef - y) ** 2).sum())
            if sse < best[0]:
                best = (sse, (float(k), float(coef[0]), float(coef[1])))

    solve(np.geomspace(math.pi / (800 * span), math.pi / (1.0 * span), 400))
    k0 = best[1][0]
    solve(np.geomspace(k0 / 1.05, k0 * 1.05, 300))
    k, a, b = best[1]
    rmse = math.sqrt(best[0] / len(t))
    return {"k": k, "a": a, "b": b, "rmse": rmse, "rmse_rel": rmse / peak,
            "total_rel": math.pi / k, "lr_min": a - b, "lr_min_ratio": (a - b) / peak}


def cosine_end_for_min_ratio(t, y, ratio: float):
    """Fin de cosinus supposant lr_min = ratio * lr(pic) : renvoie (T relatif, rmse relatif)."""
    y0 = float(y[0])
    a = y0 * (1 + ratio) / 2
    b = y0 * (1 - ratio) / 2
    span = float(t.max())
    ks = np.geomspace(math.pi / (800 * span), math.pi / (1.0 * span), 2000)
    sse = np.array([((a + b * np.cos(k * t) - y) ** 2).sum() for k in ks])
    k = float(ks[int(np.argmin(sse))])
    return math.pi / k, math.sqrt(float(sse.min()) / len(t)) / y0


# --------------------------------------------------------------------------- chargement
def load_raw(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Benchmark introuvable: {csv_path}")
    raw = pd.read_csv(csv_path)
    if raw.empty:
        raise ValueError("Le benchmark est vide")
    missing = set(REQUIRED_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans le benchmark: {sorted(missing)}")
    for col in raw.columns:
        if col != "event":
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
    if "event" not in raw:
        raw["event"] = "train"
    raw["event"] = raw["event"].fillna("train").astype(str)
    for col in ("val_loss", "timestamp"):
        if col not in raw:
            raw[col] = np.nan
    raw["row"] = np.arange(len(raw))
    return raw


def clean_data(raw: pd.DataFrame):
    """Sépare train/validation, retire les valeurs non finies, gère les reprises après crash."""
    info: dict = {"rows_raw": int(len(raw))}
    val = raw[(raw["event"] == "validation") & raw["val_loss"].notna()].copy()
    train = raw[raw["event"] == "train"].copy()
    finite = np.isfinite(train[TRAIN_NEEDED].to_numpy(float)).all(axis=1)
    info["dropped_nonfinite"] = int((~finite).sum())
    info["first_nonfinite_step"] = (
        int(train.loc[~finite, "step"].dropna().min()) if (~finite).any() and train.loc[~finite, "step"].notna().any() else None
    )
    train = train[finite].copy()
    if train.empty:
        raise ValueError("Aucune ligne d'entraînement exploitable")

    train["run_id"] = (train["step"].diff() <= 0).cumsum().astype(int)
    first_step = train.groupby("run_id")["step"].first().to_dict()
    n_runs = len(first_step)
    cutoff = {k: min([first_step[j] for j in range(k + 1, n_runs)], default=np.inf) for k in range(n_runs)}
    train["superseded"] = train["run_id"].map(cutoff).to_numpy(float) <= train["step"].to_numpy(float)

    lost = train[train["superseded"]].copy()
    clean = train[~train["superseded"]].copy().reset_index(drop=True)
    info.update(rows_train=int(len(train)), rows_validation=int(len(val)),
                rows_superseded=int(len(lost)), rows_clean=int(len(clean)), n_runs=int(n_runs))

    interval = float(np.median(np.diff(clean["step"]))) if len(clean) > 1 else 1.0
    interval = interval if interval > 0 else 1.0

    restarts = []
    for k in range(1, n_runs):
        rows_k = train[train["run_id"] == k]
        rows_prev = train[train["run_id"] == k - 1]
        if rows_k.empty or rows_prev.empty:
            continue
        gap = float(rows_k["timestamp"].iloc[0] - rows_prev["timestamp"].iloc[-1] - rows_k["elapsed_seconds"].iloc[0])
        restarts.append({
            "run": k,
            "last_step_before": int(rows_prev["step"].iloc[-1]),
            "resumed_step": int(rows_k["step"].iloc[0]),
            "replayed_rows": int(((train["run_id"] == k - 1) & train["superseded"]).sum()),
            "downtime_s": gap if math.isfinite(gap) else None,
        })

    resume = None
    if len(lost):
        merged = lost.merge(clean, on="step", suffixes=("_old", "_new"))
        if len(merged) >= 3:
            d_loss = merged["loss_new"] - merged["loss_old"]
            resume = {
                "overlap_steps": int(len(merged)),
                "step_range": [int(merged["step"].min()), int(merged["step"].max())],
                "mean_delta_loss": float(d_loss.mean()),
                "rms_delta_loss": float(math.sqrt((d_loss**2).mean())),
                "loss_corr": float(np.corrcoef(merged["loss_old"], merged["loss_new"])[0, 1]),
                "max_abs_delta_lr": float((merged["learning_rate_new"] - merged["learning_rate_old"]).abs().max()),
                "mean_delta_grad": float((merged["grad_norm_new"] - merged["grad_norm_old"]).mean()),
            }
    return clean, val, lost, restarts, resume, info, interval


# --------------------------------------------------------------------------- analyse
def analyze(raw: pd.DataFrame, args) -> SimpleNamespace:
    df, val, lost, restarts, resume, info, interval = clean_data(raw)
    M: dict = {"data": info, "restarts": restarts, "resume_check": resume}
    n = len(df)
    x = df["step"].to_numpy(float)
    loss = df["loss"].to_numpy(float)
    lr = df["learning_rate"].to_numpy(float)
    grad = df["grad_norm"].to_numpy(float)
    el = df["elapsed_seconds"].to_numpy(float)
    tps = df["tokens_per_second"].to_numpy(float)

    # ---- tokens / temps
    tokens_interval = tps * el
    df["tokens_interval"] = tokens_interval
    df["cum_tokens"] = np.cumsum(tokens_interval)
    df["active_time"] = np.cumsum(el)
    ts = df["timestamp"].to_numpy(float)
    has_ts = bool(np.isfinite(ts).all())
    df["wall_time"] = ts - (ts[0] - el[0]) if has_ts else np.nan
    tokens_per_step = float(np.median(tokens_interval) / interval)
    accumulation = float(np.median(np.diff(df["batch"]) / np.diff(df["step"]))) if n > 1 else float("nan")
    M["run"] = {
        "first_step": int(x[0]), "last_step": int(x[-1]), "log_interval_steps": interval,
        "epochs": [int(df["epoch"].min()), int(df["epoch"].max())],
        "tokens_per_step": tokens_per_step, "tokens_per_interval_cv": float(np.std(tokens_interval) / np.mean(tokens_interval)),
        "microbatches_per_step": accumulation,
        "total_tokens": float(df["cum_tokens"].iloc[-1]),
        "active_time_s": float(el.sum()),
    }

    # ---- tendances / bruit de la loss
    tw = args.trend_window
    trend, _ = robust_trend(x, loss, tw, 2)
    slope_win = min(4 * tw + 1, n if n % 2 else n - 1)
    _, d1_wide = local_poly(x, trend, slope_win, 2)
    res = loss - trend
    scale = (pd.Series(np.abs(res)).rolling(101, center=True, min_periods=20).median() * 1.4826).to_numpy()
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, robust_sigma(res) or 1e-9)
    z = res / scale
    df["loss_ema_fast"] = ema(loss, args.ema_fast)
    df["loss_ema_slow"] = ema(loss, args.ema_slow)
    df["loss_trend"] = trend
    df["loss_res"] = res
    df["loss_scale"] = scale
    df["loss_z"] = z
    df["loss_improve_per_1k"] = -d1_wide * 1000.0
    df["loss_spike"] = np.where(z > args.z, 1, np.where(z < -args.z, -1, 0))
    df["ppl"] = np.exp(np.clip(df["loss_ema_fast"], -50, 50))

    g_log = np.log(grad)
    g_trend, _ = robust_trend(x, g_log, tw, 2)
    g_res = g_log - g_trend
    g_scale = (pd.Series(np.abs(g_res)).rolling(101, center=True, min_periods=20).median() * 1.4826).to_numpy()
    g_scale = np.where(np.isfinite(g_scale) & (g_scale > 0), g_scale, robust_sigma(g_res) or 1e-9)
    df["grad_ema"] = ema(grad, args.ema_fast)
    df["grad_trend"] = np.exp(g_trend)
    df["grad_res"] = g_res
    df["grad_z"] = g_res / g_scale
    df["grad_spike"] = np.where(df["grad_z"] > args.z, 1, 0)

    warm_idx = int(np.argmax(lr))
    burn = int(min(max(tw // 2, 5), n // 4))

    # ---- loss : résumé
    first_k = min(3, n)
    ema_slow = df["loss_ema_slow"].to_numpy()
    best_trend_idx = int(np.argmin(trend))
    best_raw_idx = int(np.argmin(loss))
    third = max(n // 3, 1)
    noise = {
        "early": float(np.std(res[burn:third])) if third > burn + 2 else float("nan"),
        "mid": float(np.std(res[third:2 * third])),
        "late": float(np.std(res[2 * third:])),
    }
    windows = []
    for w_steps in (500, 1000, 2000, 5000):
        sel = x >= x[-1] - w_steps
        if sel.sum() >= 20 and w_steps < (x[-1] - x[0]):
            o = ols(x[sel], loss[sel])
            gain = -o["slope"] * 1000
            ci = 1.96 * o["se"] * 1000
            windows.append({
                "window_steps": w_steps, "gain_per_1k_steps": gain, "ci95": ci,
                "gain_per_100m_tokens": -o["slope"] / tokens_per_step * 1e8,
                "significant": bool(gain - ci > 0),
            })
    prog_total = ema_slow[0] - ema_slow[-1]
    milestones = []
    if prog_total > 0:
        for frac in (0.5, 0.75, 0.9, 0.95, 0.99):
            hit = np.nonzero(ema_slow <= ema_slow[0] - frac * prog_total)[0]
            if len(hit):
                i = int(hit[0])
                milestones.append({"fraction": frac, "step": int(x[i]), "loss": float(ema_slow[i]),
                                   "tokens_m": float(df["cum_tokens"].iloc[i] / 1e6)})
    M["loss"] = {
        "initial": float(loss[:first_k].mean()), "final_trend": float(trend[-1]),
        "final_ema_fast": float(df["loss_ema_fast"].iloc[-1]), "final_raw_mean_last_100": float(loss[-100:].mean()),
        "best_raw": {"step": int(x[best_raw_idx]), "loss": float(loss[best_raw_idx])},
        "best_trend": {"step": int(x[best_trend_idx]), "loss": float(trend[best_trend_idx])},
        "noise_sigma": noise, "windows": windows, "milestones": milestones,
        "perplexity_final": float(math.exp(min(trend[-1], 50))),
        "bits_per_token_final": float(trend[-1] / math.log(2)),
        "n_spikes_up": int((df["loss_spike"] == 1).sum()), "n_spikes_down": int((df["loss_spike"] == -1).sum()),
    }
    if windows:
        last_w = windows[min(2, len(windows) - 1)]
        M["loss"]["plateau_verdict"] = (
            "amélioration encore significative" if last_w["significant"] else "plateau probable (pente non significative)"
        )
        M["loss"]["plateau_window"] = last_w["window_steps"]

    # ---- loi de puissance + backtest + extrapolation
    fit_start = max(x[warm_idx], np.quantile(x, 0.05))
    fmask = x >= fit_start
    if fmask.sum() < 30:
        fmask = np.arange(n) >= int(0.3 * n)
    pl = bt = None
    extrap = []
    try:
        pl = fit_power_law(x[fmask], loss[fmask])
        xs, ys = x[fmask], loss[fmask]
        cut = int(0.7 * len(xs))
        if cut >= 20 and len(xs) - cut >= 10:
            pl_bt = fit_power_law(xs[:cut], ys[:cut])
            pred = power_law(xs[cut:], pl_bt)
            bt = {"params": pl_bt, "oos_rmse": float(np.sqrt(np.mean((pred - ys[cut:]) ** 2))),
                  "oos_bias": float(np.mean(pred - ys[cut:]))}
        for factor in (1.5, 2.0, 4.0):
            s_h = x[-1] * factor
            extrap.append({"factor": factor, "step": int(s_h), "loss": float(power_law(s_h, pl)),
                           "tokens_b": float(s_h * tokens_per_step / 1e9)})
    except Exception as exc:  # ajustement impossible sur des données dégénérées
        M["loss"]["power_law_error"] = str(exc)
    M["loss"]["power_law"] = {**pl, "fit_start_step": int(fit_start)} if pl else None
    M["loss"]["backtest"] = bt
    M["loss"]["extrapolation"] = extrap

    # ---- résidus : normalité, autocorrélation, périodogramme
    zz = z[burn:]
    skew = float(np.mean(zz**3) / np.std(zz) ** 3) if np.std(zz) > 0 else float("nan")
    kurt = float(np.mean(zz**4) / np.std(zz) ** 4 - 3) if np.std(zz) > 0 else float("nan")
    jb = len(zz) / 6.0 * (skew**2 + kurt**2 / 4.0)
    rr = res[burn:] - res[burn:].mean()
    max_lag = min(60, len(rr) // 3)
    acf = np.array([1.0] + [float((rr[:-k] * rr[k:]).sum() / (rr * rr).sum()) for k in range(1, max_lag + 1)])
    r1 = acf[1] if len(acf) > 1 else 0.0
    n_eff = len(rr) * (1 - r1) / (1 + r1) if r1 > -0.99 else len(rr)
    win = np.hanning(len(rr))
    power = np.abs(np.fft.rfft(rr * win)) ** 2
    freqs = np.fft.rfftfreq(len(rr), d=interval)  # cycles / step
    valid = np.arange(len(power)) >= 3
    med_power = float(np.median(power[valid]))
    peaks = []
    for idx in np.argsort(power * valid)[::-1][:3]:
        if valid[idx] and freqs[idx] > 0:
            peaks.append({"period_steps": float(1 / freqs[idx]), "ratio_to_median": float(power[idx] / med_power),
                          "significant": bool(power[idx] / med_power > 15)})
    M["residuals"] = {
        "skewness": skew, "excess_kurtosis": kurt, "jarque_bera": float(jb), "jb_p": float(math.exp(-jb / 2)),
        "acf_lag1": float(r1), "acf_lag2": float(acf[2]) if len(acf) > 2 else None, "n_effective": float(n_eff), "n": int(len(rr)),
        "periodogram_peaks": peaks,
    }
    df.attrs["acf"] = acf
    df.attrs["periodogram"] = (freqs, power, med_power, interval)

    # ---- gradient
    gmask_late = x >= x[0] + 0.5 * (x[-1] - x[0])
    og = ols(x[gmask_late], grad[gmask_late])
    max_idx = int(np.argmax(grad))
    M["grad"] = {
        "median": float(np.median(grad)), "p95": float(np.quantile(grad, 0.95)), "p99": float(np.quantile(grad, 0.99)),
        "max": float(grad[max_idx]), "max_step": int(x[max_idx]),
        "early_median": float(np.median(grad[: max(n // 10, 3)])), "late_median": float(np.median(grad[-max(n // 10, 3):])),
        "late_trend_per_1k_steps": float(og["slope"] * 1000), "late_trend_ci95": float(1.96 * og["se"] * 1000),
        "n_spikes": int(df["grad_spike"].sum()),
        "corr_loss_res_vs_grad_res": float(np.corrcoef(res[burn:], g_res[burn:])[0, 1]),
        "clip_fraction": float(np.mean(grad > args.grad_clip)) if args.grad_clip else None,
        "clip_threshold": args.grad_clip,
    }

    # ---- learning rate
    peak = float(lr[warm_idx])
    lr_info: dict = {"peak": peak, "peak_step": int(x[warm_idx]), "final": float(lr[-1]), "final_over_peak": float(lr[-1] / peak)}
    sched = {"model": "constant" if np.ptp(lr) < 1e-9 * peak else "inconnu"}
    if warm_idx >= 4:
        ow = ols(x[: warm_idx + 1], lr[: warm_idx + 1])
        lr_info["warmup"] = {"steps_estimate": float((peak - ow["intercept"]) / ow["slope"]), "linear_r2": float(ow["r2"])}
    t_rel = x[warm_idx:] - x[warm_idx]
    y_dec = lr[warm_idx:]
    if len(t_rel) >= 12 and np.ptp(y_dec) > 1e-9 * peak:
        cos = fit_cosine_decay(t_rel, y_dec, peak)
        lin = ols(t_rel, y_dec)
        lin_rmse = float(math.sqrt(np.mean((lin["intercept"] + lin["slope"] * t_rel - y_dec) ** 2))) / peak
        end0, rm0 = cosine_end_for_min_ratio(t_rel, y_dec, 0.0)
        end10, rm10 = cosine_end_for_min_ratio(t_rel, y_dec, 0.1)
        sched = {
            "model": "cosinus" if cos["rmse_rel"] < lin_rmse / 3 else "linéaire/autre",
            "cosine": cos, "linear_rmse_rel": lin_rmse,
            "cosine_end_if_min0": {"end_step": float(x[warm_idx] + end0), "rmse_rel": rm0},
            "cosine_end_if_min10pct": {"end_step": float(x[warm_idx] + end10), "rmse_rel": rm10},
            "cosine_end_fit": float(x[warm_idx] + cos["total_rel"]),
        }
        if lin["slope"] < 0:
            sched["linear_end_step_at_zero"] = float(x[warm_idx] + y_dec[0] / -lin["slope"])
    lr_info["schedule"] = sched
    M["lr"] = lr_info
    df.attrs["warm_idx"] = warm_idx

    # ---- débit, intervalles lents, surcoûts
    med_el = float(np.median(el))
    sigma_el = max(robust_sigma(el), 0.002 * med_el)
    thresh = med_el + max(6 * sigma_el, 0.05 * med_el)
    slow = el > thresh
    first_rows = set(df.groupby("run_id").head(1).index.tolist())
    startup = np.array([i in first_rows for i in range(n)]) & slow
    cand_steps = x[slow & ~startup]
    period = phase = None
    if len(cand_steps) >= 3:
        vals, counts = np.unique(np.diff(cand_steps), return_counts=True)
        p = int(vals[int(np.argmax(counts))])
        if counts.max() >= 2 and p > 0:
            period, phase = p, int(cand_steps[0] % p)
    periodic = slow & ~startup & ((x % period == phase) if period else np.zeros(n, bool))
    isolated = slow & ~startup & ~periodic
    kind = np.full(n, "steady", dtype=object)
    kind[startup] = "startup"
    kind[periodic] = "periodic_overhead"
    kind[isolated] = "isolated_slow"
    df["interval_kind"] = kind
    steady = kind == "steady"
    df["tps_steady"] = np.where(steady, tps, np.nan)
    steady_el = float(np.median(el[steady])) if steady.any() else med_el
    excess = np.clip(el - steady_el, 0, None)
    od = ols(x[steady], tps[steady])
    tps_med = float(np.median(tps[steady]))
    k_edge = max(int(0.2 * steady.sum()), 5)
    stp = tps[steady]
    total_tokens = float(tokens_interval.sum())
    eff_tps = total_tokens / float(el.sum())
    lost_time = float(lost["elapsed_seconds"].sum()) if len(lost) else 0.0
    gaps = float(sum(r["downtime_s"] or 0.0 for r in restarts))
    budget = {
        "compute_stable_s": float(np.minimum(el, steady_el).sum()),
        "periodic_overhead_s": float(excess[periodic].sum()),
        "startup_overhead_s": float(excess[startup].sum()),
        "isolated_slow_s": float(excess[isolated].sum()),
        "crash_replayed_s": lost_time,
        "restart_downtime_s": gaps,
    }
    budget["total_s"] = float(sum(budget.values()))
    sec_per_step = float(el.sum() / (n * interval))
    total_steps = args.total_steps
    total_src = "paramètre --total-steps" if total_steps else None
    if not total_steps and sched.get("model") == "cosinus" and "cosine_end_if_min10pct" in sched:
        total_steps = int(round(sched["cosine_end_if_min10pct"]["end_step"], -2))
        total_src = "fin de cosinus estimée (lr_min = 10 % du pic)"
    eta = None
    if total_steps and total_steps > x[-1]:
        remaining = (total_steps - x[-1]) * sec_per_step
        eta = {"total_steps": int(total_steps), "source": total_src, "remaining_s": float(remaining),
               "remaining_tokens_b": float((total_steps - x[-1]) * tokens_per_step / 1e9)}
    M["timing"] = {
        "steady_median_interval_s": steady_el, "steady_tps_median": tps_med, "steady_tps_mean": float(stp.mean()),
        "steady_tps_cv": float(stp.std() / stp.mean()), "steady_tps_p01": float(np.quantile(stp, 0.01)),
        "effective_tps": eff_tps, "efficiency": float(eff_tps / tps_med),
        "tps_drift_pct_per_1k_steps": float(od["slope"] * 1000 / tps_med * 100),
        "tps_drift_ci95_pct": float(1.96 * od["se"] * 1000 / tps_med * 100),
        "tps_drift_significant": bool(abs(od["slope"]) > 2 * od["se"]),
        "tps_early_vs_late_pct": float((np.median(stp[-k_edge:]) / np.median(stp[:k_edge]) - 1) * 100),
        "slow_threshold_s": float(thresh), "n_slow": int(slow.sum()), "n_startup": int(startup.sum()),
        "n_periodic": int(periodic.sum()), "n_isolated": int(isolated.sum()),
        "periodic_every_steps": period,
        "periodic_mean_excess_s": float(excess[periodic].mean()) if periodic.any() else None,
        "sec_per_step_incl_overhead": sec_per_step, "budget": budget, "eta": eta,
        "wall_clock_s": float(df["wall_time"].iloc[-1]) if has_ts else None,
    }

    # ---- corrélations (sur le régime stable, hors phase initiale)
    mask = steady & (np.arange(n) >= burn)
    tps_res = tps - pd.Series(tps).rolling(51, center=True, min_periods=5).median().to_numpy()
    comp = pd.DataFrame({
        "loss_res": res, "grad_res": g_res, "tps_res": tps_res, "elapsed_res": el - steady_el,
        "lr": lr,
    })[mask]
    comp["lr"] = comp["lr"].diff().fillna(0)  # variation du LR plutôt que sa valeur
    corr = comp.drop(columns="lr").rank().corr()
    M["correlations_spearman"] = {a: {b: float(corr.loc[a, b]) for b in corr.columns} for a in corr.index}
    df.attrs["corr"] = corr
    lags = np.arange(-20, 21)
    a_, b_ = res[burn:], g_res[burn:]
    xc = []
    for k in lags:
        if k >= 0:
            xc.append(np.corrcoef(a_[: len(a_) - k], b_[k:])[0, 1])
        else:
            xc.append(np.corrcoef(a_[-k:], b_[: len(b_) + k])[0, 1])
    df.attrs["xcorr"] = (lags, np.array(xc), 1.96 / math.sqrt(len(a_)))

    # ---- segments (par epoch si plusieurs, sinon déciles)
    seg_rows = []
    if df["epoch"].nunique() > 1:
        groups = [(f"epoch {int(e)}", g.index.to_numpy()) for e, g in df.groupby("epoch")]
    else:
        n_seg = max(2, min(args.segments, n // 20))
        groups = [(f"{i + 1}/{n_seg}", idx) for i, idx in enumerate(np.array_split(np.arange(n), n_seg))]
    for label, idx in groups:
        sub = df.loc[idx]
        o = ols(sub["step"], sub["loss"])
        seg_rows.append({
            "segment": label, "step_start": int(sub["step"].iloc[0]), "step_end": int(sub["step"].iloc[-1]),
            "loss_mean": float(sub["loss"].mean()), "loss_end_trend": float(sub["loss_trend"].iloc[-1]),
            "loss_gain_per_1k_steps": float(-o["slope"] * 1000), "loss_noise_sigma": float(sub["loss_res"].std()),
            "grad_median": float(sub["grad_norm"].median()), "grad_p95": float(sub["grad_norm"].quantile(0.95)),
            "lr_mean": float(sub["learning_rate"].mean()), "tps_median": float(sub["tokens_per_second"].median()),
            "overhead_pct": float(100 * sub["elapsed_seconds"].sub(steady_el).clip(lower=0).sum() / sub["elapsed_seconds"].sum()),
            "spikes_loss": int((sub["loss_spike"] == 1).sum()), "spikes_grad": int(sub["grad_spike"].sum()),
        })
    segments = pd.DataFrame(seg_rows)

    # ---- anomalies
    anomalies = []
    for i in range(n):
        kinds = []
        if df["loss_spike"].iat[i] == 1:
            kinds.append("loss_spike")
        if df["loss_spike"].iat[i] == -1:
            kinds.append("loss_drop")
        if df["grad_spike"].iat[i] == 1:
            kinds.append("grad_spike")
        if kind[i] in ("startup", "isolated_slow"):
            kinds.append(kind[i])
        if kinds and i >= burn or (kinds and kind[i] in ("startup", "isolated_slow")):
            anomalies.append({
                "step": int(x[i]), "run_id": int(df["run_id"].iat[i]), "kinds": "+".join(kinds),
                "loss": float(loss[i]), "loss_z": float(z[i]), "grad_norm": float(grad[i]), "grad_z": float(df["grad_z"].iat[i]),
                "elapsed_seconds": float(el[i]), "tokens_per_second": float(tps[i]),
            })
    anomalies = pd.DataFrame(anomalies, columns=[
        "step", "run_id", "kinds", "loss", "loss_z", "grad_norm", "grad_z", "elapsed_seconds", "tokens_per_second"])
    M["anomalies"] = {"count": int(len(anomalies))}

    # ---- validation (si présente)
    if len(val):
        vs = val.sort_values("step")
        M["validation"] = {"n": int(len(vs)), "best": {"step": int(vs.loc[vs["val_loss"].idxmin(), "step"]),
                                                          "val_loss": float(vs["val_loss"].min())},
                           "final": float(vs["val_loss"].iloc[-1])}
        # écart de généralisation : val_loss - tendance train interpolée au même step
        gap = vs["val_loss"].to_numpy(float) - np.interp(vs["step"].to_numpy(float), x, trend)
        M["validation"]["gap_final"] = float(gap[-1])
        M["validation"]["gap_trend_per_1k_steps"] = float(ols(vs["step"], gap)["slope"] * 1000) if len(vs) >= 4 else None

    return SimpleNamespace(df=df, val=val, lost=lost, restarts=restarts, M=M, segments=segments, anomalies=anomalies,
                           interval=interval, warm_idx=warm_idx, burn=burn, pl=pl, bt=bt, args=args,
                           tokens_per_step=tokens_per_step, steady_el=steady_el)


# --------------------------------------------------------------------------- rapport
def fmt_dur(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 90:
        return f"{seconds:.1f} s"
    if seconds < 5400:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.2f} h"


def build_report(A: SimpleNamespace, csv_path: Path) -> list[str]:
    M = A.M
    d, r, L, G, R, T = M["data"], M["run"], M["loss"], M["grad"], M["lr"], M["timing"]
    out = [f"# Analyse du benchmark d'entraînement", f"Source : `{csv_path}`", ""]

    out += ["## Données",
            f"- {d['rows_raw']:,} lignes lues, {d['rows_clean']:,} retenues pour l'analyse "
            f"(steps {r['first_step']:,} → {r['last_step']:,}, un point tous les {r['log_interval_steps']:.0f} steps).",
            f"- Epochs : {r['epochs'][0]} → {r['epochs'][1]} | validations : {d['rows_validation']}",
            f"- {r['tokens_per_step']:,.0f} tokens/step ({r['microbatches_per_step']:.1f} micro-batch(es)/step), "
            f"{r['total_tokens'] / 1e6:,.1f} M tokens traités, {fmt_dur(r['active_time_s'])} de calcul."]
    if d["dropped_nonfinite"]:
        out.append(f"- ATTENTION : {d['dropped_nonfinite']} ligne(s) non finie(s) retirée(s) "
                   f"(premier step concerné : {d['first_nonfinite_step']}) - signe possible de divergence.")
    for rs in M["restarts"]:
        down = f", {fmt_dur(rs['downtime_s'])} d'arrêt" if rs["downtime_s"] is not None else ""
        out.append(f"- Redémarrage détecté : le step est passé de {rs['last_step_before']:,} à {rs['resumed_step']:,} "
                   f"({rs['replayed_rows']} mesures rejouées, ignorées de l'analyse{down}).")
    rc = M["resume_check"]
    if rc:
        verdict = "reprise fidèle" if rc["rms_delta_loss"] < 2 * L["noise_sigma"]["mid"] and rc["loss_corr"] > 0.99 else \
            "trajectoire différente après reprise (état optimiseur/données non identique ?)"
        out.append(f"- Cohérence de la reprise sur {rc['overlap_steps']} steps communs : Δloss moyen {rc['mean_delta_loss']:+.4f}, "
                   f"RMS {rc['rms_delta_loss']:.4f}, corrélation {rc['loss_corr']:.4f}, Δlr max {rc['max_abs_delta_lr']:.1e} → {verdict}.")

    out += ["", "## Loss",
            f"- {L['initial']:.3f} → {L['final_trend']:.3f} (tendance finale) ; moyenne des 100 dernières mesures {L['final_raw_mean_last_100']:.3f}.",
            f"- Meilleure valeur brute {L['best_raw']['loss']:.3f} (step {L['best_raw']['step']:,}) ; meilleure tendance "
            f"{L['best_trend']['loss']:.3f} (step {L['best_trend']['step']:,}).",
            f"- Perplexité ≈ {L['perplexity_final']:.1f}, {L['bits_per_token_final']:.3f} bit/token (si loss = entropie croisée en nats).",
            f"- Bruit de mesure (σ des résidus) : début {L['noise_sigma']['early']:.3f}, milieu {L['noise_sigma']['mid']:.3f}, fin {L['noise_sigma']['late']:.3f}.",
            f"- Pics : {L['n_spikes_up']} hausses et {L['n_spikes_down']} chutes au-delà de {A.args.z:g} σ robustes."]
    if L["windows"]:
        out.append("- Gain de loss sur les derniers steps :")
        for w in L["windows"]:
            flag = "significatif" if w["significant"] else "non significatif"
            out.append(f"    - {w['window_steps']:,} derniers steps : {w['gain_per_1k_steps']:+.4f} ± {w['ci95']:.4f} /1k steps "
                       f"({w['gain_per_100m_tokens']:+.4f} /100 M tokens) - {flag}")
        out.append(f"- Diagnostic : {L['plateau_verdict']} (fenêtre {L['plateau_window']:,} steps).")
    if L["milestones"]:
        out.append("- Jalons de progression (part de l'amélioration totale atteinte) : " + " ; ".join(
            f"{int(m['fraction'] * 100)} % → step {m['step']:,} ({m['tokens_m']:,.0f} M tok, loss {m['loss']:.3f})" for m in L["milestones"]))
    pl = L.get("power_law")
    if pl:
        out.append(f"- Loi de puissance (steps ≥ {pl['fit_start_step']:,}) : loss ≈ {pl['l_inf']:.3f} + {pl['A']:.2f}·step^-{pl['b']:.3f} "
                   f"(R² {pl['r2']:.3f}, RMSE {pl['rmse']:.4f})" + (" - borne atteinte, L_inf peu fiable" if pl["at_bound"] else ""))
        if L["backtest"]:
            out.append(f"    - Backtest (ajustement sur 70 % puis prédiction des 30 % suivants) : RMSE {L['backtest']['oos_rmse']:.4f}, "
                       f"biais {L['backtest']['oos_bias']:+.4f}.")
        if L["extrapolation"]:
            out.append("    - Extrapolation à LR courant (indicative, ignore le gain d'un decay) : " + " ; ".join(
                f"step {e['step']:,} → {e['loss']:.3f}" for e in L["extrapolation"]))
    if "validation" in M:
        v = M["validation"]
        out.append(f"- Validation : {v['n']} mesures, meilleure {v['best']['val_loss']:.4f} (step {v['best']['step']:,}), "
                   f"écart val - train final {v['gap_final']:+.4f}.")

    out += ["", "## Gradient",
            f"- Norme : médiane {G['median']:.3f}, p95 {G['p95']:.3f}, p99 {G['p99']:.3f}, max {G['max']:.3f} (step {G['max_step']:,}).",
            f"- Début {G['early_median']:.3f} → fin {G['late_median']:.3f} (médianes sur 10 % des points) ; "
            f"tendance sur la 2e moitié : {G['late_trend_per_1k_steps']:+.4f} ± {G['late_trend_ci95']:.4f} /1k steps.",
            f"- {G['n_spikes']} pic(s) de gradient ; corrélation résidus loss ↔ gradient : {G['corr_loss_res_vs_grad_res']:+.2f}."]
    if G["clip_fraction"] is not None:
        out.append(f"- {100 * G['clip_fraction']:.1f} % des mesures dépassent le seuil de clipping {G['clip_threshold']:g}.")

    out += ["", "## Learning rate",
            f"- Pic {R['peak']:.3e} au step {R['peak_step']:,} ; valeur finale {R['final']:.3e} ({100 * R['final_over_peak']:.1f} % du pic)."]
    if "warmup" in R:
        out.append(f"- Warmup linéaire d'environ {R['warmup']['steps_estimate']:.0f} steps (R² {R['warmup']['linear_r2']:.4f}).")
    s = R["schedule"]
    if s["model"] == "cosinus":
        c = s["cosine"]
        out.append(f"- Décroissance de type cosinus (erreur relative {c['rmse_rel']:.1e} contre {s['linear_rmse_rel']:.1e} pour une droite) ; "
                   f"fin de schedule estimée vers le step {s['cosine_end_if_min10pct']['end_step']:,.0f} (lr_min = 10 % du pic) "
                   f"ou {s['cosine_end_if_min0']['end_step']:,.0f} (lr_min = 0).")
    elif s["model"] == "constant":
        out.append("- Learning rate constant.")
    else:
        out.append("- Forme de décroissance non reconnue comme cosinus.")

    out += ["", "## Débit et temps",
            f"- Régime stable : {T['steady_tps_median']:,.0f} tokens/s (CV {100 * T['steady_tps_cv']:.2f} %), "
            f"débit effectif {T['effective_tps']:,.0f} tokens/s soit {100 * T['efficiency']:.1f} % du régime stable.",
            f"- Dérive du débit stable : {T['tps_drift_pct_per_1k_steps']:+.3f} ± {T['tps_drift_ci95_pct']:.3f} %/1k steps "
            f"({'significative' if T['tps_drift_significant'] else 'non significative'}) ; "
            f"début → fin : {T['tps_early_vs_late_pct']:+.2f} %."]
    if T["periodic_every_steps"]:
        out.append(f"- Surcoût périodique tous les {T['periodic_every_steps']:,} steps ({T['n_periodic']} occurrences, "
                   f"+{T['periodic_mean_excess_s']:.2f} s chacune : checkpoint/éval probable).")
    out.append(f"- Intervalles lents (> {T['slow_threshold_s']:.2f} s) : {T['n_slow']} dont {T['n_startup']} démarrage(s), "
               f"{T['n_periodic']} périodique(s), {T['n_isolated']} isolé(s).")
    b = T["budget"]
    tot = b["total_s"] or 1.0
    out.append("- Budget temps : " + ", ".join(
        f"{label} {fmt_dur(b[key])} ({100 * b[key] / tot:.1f} %)" for label, key in [
            ("calcul stable", "compute_stable_s"), ("surcoût périodique", "periodic_overhead_s"),
            ("démarrages", "startup_overhead_s"), ("lents isolés", "isolated_slow_s"),
            ("crash rejoué", "crash_replayed_s"), ("arrêt de redémarrage", "restart_downtime_s")] if b[key] > 0))
    if T["eta"]:
        e = T["eta"]
        out.append(f"- ETA jusqu'au step {e['total_steps']:,} ({e['source']}) : {fmt_dur(e['remaining_s'])} "
                   f"({e['remaining_tokens_b']:.2f} G tokens restants, {T['sec_per_step_incl_overhead']:.3f} s/step surcoûts inclus).")

    S = M["residuals"]
    out += ["", "## Statistiques des résidus de loss",
            f"- Asymétrie {S['skewness']:+.2f}, kurtosis en excès {S['excess_kurtosis']:+.2f}, Jarque-Bera p ≈ {S['jb_p']:.2g} "
            f"({'résidus non gaussiens (queues lourdes / pics)' if S['jb_p'] < 0.01 else 'compatibles avec un bruit gaussien'}).",
            f"- Autocorrélation lag 1 = {S['acf_lag1']:+.2f} → {S['n_effective']:.0f} observations indépendantes sur {S['n']}."]
    sig = [p for p in S["periodogram_peaks"] if p["significant"]]
    if sig:
        out.append("- Périodicité significative dans les résidus : " + ", ".join(
            f"~{p['period_steps']:,.0f} steps ({p['ratio_to_median']:.0f}× la médiane)" for p in sig))
    else:
        out.append("- Aucune périodicité significative dans les résidus (pic max "
                   f"{S['periodogram_peaks'][0]['ratio_to_median']:.1f}× la médiane, seuil 15×).")
    cs = M["correlations_spearman"]
    out.append(f"- Corrélations de Spearman (régime stable) : loss↔grad {cs['loss_res']['grad_res']:+.2f}, "
               f"loss↔débit {cs['loss_res']['tps_res']:+.2f}, grad↔débit {cs['grad_res']['tps_res']:+.2f}.")
    out.append(f"- Anomalies listées : {M['anomalies']['count']} (voir benchmark_anomalies.csv).")
    return out


def summary_panel(A: SimpleNamespace) -> list[str]:
    M = A.M
    r, L, G, T, R = M["run"], M["loss"], M["grad"], M["timing"], M["lr"]
    lines = [
        "SUMMARY",
        f"Steps        {r['first_step']:,} → {r['last_step']:,}",
        f"Tokens       {r['total_tokens'] / 1e6:,.0f} M",
        f"Loss         {L['initial']:.3f} → {L['final_trend']:.3f}",
        f"Best trend   {L['best_trend']['loss']:.3f} @ {L['best_trend']['step']:,}",
        f"Noise σ      {L['noise_sigma']['late']:.3f} (late)",
    ]
    if L["windows"]:
        w = L["windows"][min(2, len(L["windows"]) - 1)]
        lines.append(f"Gain /1k st  {w['gain_per_1k_steps']:+.4f} ({w['window_steps']:,} st)")
    if L.get("power_law"):
        lines.append(f"L_inf (fit)  {L['power_law']['l_inf']:.3f}")
    lines += [
        f"Grad median  {G['median']:.3f}  max {G['max']:.2f}",
        f"LR peak      {R['peak']:.2e}",
        f"Tok/s stable {T['steady_tps_median']:,.0f}",
        f"Efficiency   {100 * T['efficiency']:.1f} %",
        f"Restarts     {len(A.restarts)}",
        f"Loss spikes  {L['n_spikes_up']} up / {L['n_spikes_down']} down",
    ]
    if T["eta"]:
        lines.append(f"ETA          {fmt_dur(T['eta']['remaining_s'])} → {T['eta']['total_steps']:,}")
    return lines


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return to_jsonable(obj.to_dict(orient="records"))
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


# --------------------------------------------------------------------------- figures
def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": C["surface"], "axes.facecolor": C["surface"], "savefig.facecolor": C["surface"],
        "axes.edgecolor": C["axis"], "axes.labelcolor": C["ink2"], "axes.titlecolor": C["ink"],
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.6,
        "xtick.color": C["muted"], "ytick.color": C["muted"], "text.color": C["ink"],
        "font.size": 9, "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 1.6,
    })


def style(ax, title: str, xlabel: str | None = "Optimizer step", ylabel: str | None = None) -> None:
    ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def restart_lines(ax, A, label: bool = True) -> None:
    for i, rs in enumerate(A.restarts):
        ax.axvline(rs["resumed_step"] - A.interval / 2, color=C["muted"], ls=(0, (4, 3)), lw=0.9, zorder=1,
                   label="restart" if (label and i == 0) else None)


def legend(ax, **kw) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if len([lb for lb in labels if lb and not lb.startswith("_")]) >= 2:
        ax.legend(**kw)


def thousands(ax, axis="x") -> None:
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}")
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(fmt)


def spike_scatter(ax, x, y, mask, label="spike"):
    if mask.any():
        ax.scatter(x[mask], y[mask], s=30, color=C["red"], edgecolor=C["surface"], linewidth=1, zorder=5, label=label)


def fig_overview(A, path: Path) -> None:
    df, M = A.df, A.M
    x = df["step"].to_numpy()
    fig, axes = plt.subplots(3, 2, figsize=(15, 13), constrained_layout=True)
    fig.suptitle("Training benchmark - overview", fontsize=15, fontweight="bold", x=0.01, ha="left")

    ax = axes[0, 0]
    ax.scatter(x, df["loss"], s=6, color=C["blue"], alpha=0.30, linewidth=0, label="raw")
    ax.plot(x, df["loss_ema_fast"], color=C["blue"], lw=1.4, label=f"EMA {A.args.ema_fast}")
    ax.plot(x, df["loss_ema_slow"], color=C["blue_dark"], lw=2, label=f"EMA {A.args.ema_slow}")
    if len(A.val):
        ax.scatter(A.val["step"], A.val["val_loss"], s=30, color=C["orange"], edgecolor=C["surface"], zorder=4, label="validation")
    spike_scatter(ax, x, df["loss"].to_numpy(), df["loss_spike"].to_numpy() != 0, "loss spike")
    restart_lines(ax, A)
    ax.annotate(f"{df['loss_trend'].iloc[-1]:.3f}", (x[-1], df["loss_trend"].iloc[-1]), xytext=(-4, 10),
                textcoords="offset points", ha="right", color=C["ink2"])
    style(ax, "Loss", ylabel="Loss")
    thousands(ax)
    legend(ax)

    ax = axes[0, 1]
    late = x >= x[A.warm_idx]
    sub = df[late]
    ax.scatter(sub["step"], sub["loss"], s=6, color=C["blue"], alpha=0.30, linewidth=0, label="raw")
    ax.fill_between(sub["step"], sub["loss_trend"] - 2 * sub["loss_scale"], sub["loss_trend"] + 2 * sub["loss_scale"],
                    color=C["blue"], alpha=0.14, linewidth=0, label="trend ±2σ (local)")
    ax.plot(sub["step"], sub["loss_trend"], color=C["blue_dark"], lw=2, label="robust trend")
    lo, hi = np.quantile(sub["loss"], [0.002, 0.995])
    ax.set_ylim(lo - 0.02, hi + 0.02)
    restart_lines(ax, A, label=False)
    style(ax, "Loss after warmup (zoom)", ylabel="Loss")
    thousands(ax)
    legend(ax)

    ax = axes[1, 0]
    ax.plot(x, df["learning_rate"], color=C["aqua"])
    ax.axvspan(x[0], x[A.warm_idx], color=C["aqua"], alpha=0.10, linewidth=0)
    ax.annotate("warmup", (x[A.warm_idx] / 2, df["learning_rate"].max() * 0.5), ha="center", color=C["ink2"])
    ax.scatter([x[A.warm_idx]], [df["learning_rate"].iloc[A.warm_idx]], s=40, color=C["aqua"], edgecolor=C["surface"], linewidth=1.5, zorder=5)
    restart_lines(ax, A, label=False)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    style(ax, "Learning rate", ylabel="LR")
    thousands(ax)

    ax = axes[1, 1]
    g = df["grad_norm"]
    ax.scatter(x, g, s=6, color=C["orange"], alpha=0.30, linewidth=0, label="raw")
    roll = g.rolling(A.args.window, center=True, min_periods=5)
    ax.fill_between(x, roll.quantile(0.1), roll.quantile(0.9), color=C["orange"], alpha=0.16, linewidth=0, label="rolling p10-p90")
    ax.plot(x, df["grad_ema"], color=C["orange"], lw=2, label=f"EMA {A.args.ema_fast}")
    spike_scatter(ax, x, g.to_numpy(), df["grad_spike"].to_numpy() == 1, "grad spike")
    ax.set_yscale("log")
    restart_lines(ax, A, label=False)
    style(ax, "Gradient norm", ylabel="Norm (log)")
    thousands(ax)
    legend(ax)

    ax = axes[2, 0]
    kinds = [("steady", C["violet"], "o", "steady"), ("periodic_overhead", C["red"], "X", "periodic overhead (eval / checkpoint)"),
             ("startup", C["muted"], "D", "startup / resume"), ("isolated_slow", C["yellow"], "s", "other slow")]
    for key, color, marker, label in kinds:
        m = (df["interval_kind"] == key).to_numpy()
        if m.any():
            ax.scatter(x[m], df["tokens_per_second"][m], s=8 if key == "steady" else 38, color=color, marker=marker,
                       alpha=0.5 if key == "steady" else 1, edgecolor=C["surface"] if key != "steady" else "none", linewidth=0.8, label=label)
    restart_lines(ax, A, label=False)
    style(ax, "Throughput", ylabel="Tokens / second")
    thousands(ax)
    thousands(ax, "y")
    legend(ax, loc="lower right")

    ax = axes[2, 1]
    ax.axis("off")
    ax.grid(False)
    ax.text(0.02, 0.98, "\n".join(summary_panel(A)), va="top", family="monospace", fontsize=10.5, color=C["ink"])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_loss(A, path: Path) -> None:
    df, M = A.df, A.M
    x = df["step"].to_numpy()
    fig, axes = plt.subplots(3, 2, figsize=(15, 13), constrained_layout=True)
    fig.suptitle("Loss - detailed analysis", fontsize=15, fontweight="bold", x=0.01, ha="left")

    ax = axes[0, 0]
    z = df["loss_z"].to_numpy()
    ax.scatter(x, df["loss_res"], s=6, color=C["blue"], alpha=0.35, linewidth=0, label="residual (loss - trend)")
    ax.fill_between(x, -A.args.z * df["loss_scale"], A.args.z * df["loss_scale"], color=C["blue"], alpha=0.12, linewidth=0,
                    label=f"±{A.args.z:g} robust σ (local)")
    spike_scatter(ax, x, df["loss_res"].to_numpy(), df["loss_spike"].to_numpy() != 0, "flagged")
    ax.axhline(0, color=C["axis"], lw=0.8)
    restart_lines(ax, A, label=False)
    style(ax, "Residuals and local noise band", ylabel="Residual")
    thousands(ax)
    legend(ax)

    ax = axes[0, 1]
    ax.plot(x, df["loss_scale"], color=C["blue"], label="local σ (robust)")
    ax.plot(x, df["loss_res"].rolling(A.args.window, center=True, min_periods=5).std(), color=C["blue_dark"], lw=1.2,
            label=f"rolling std ({A.args.window})")
    ax.set_yscale("log")
    restart_lines(ax, A, label=False)
    style(ax, "Noise level over training", ylabel="σ (log)")
    thousands(ax)
    legend(ax)

    ax = axes[1, 0]
    pl, bt = A.pl, A.bt
    post = x >= x[A.warm_idx]
    ax.scatter(x[post], df["loss"][post], s=6, color=C["blue"], alpha=0.30, linewidth=0, label="raw")
    ax.plot(x[post], df["loss_trend"][post], color=C["blue_dark"], lw=1.8, label="robust trend")
    if pl:
        grid = np.geomspace(max(x[A.warm_idx], 1), x[-1] * 4, 300)
        inside = grid <= x[-1]
        ax.plot(grid[inside], power_law(grid[inside], pl), color=C["orange"], lw=2, label="power-law fit")
        ax.plot(grid[~inside], power_law(grid[~inside], pl), color=C["orange"], lw=2, ls=(0, (4, 3)), label="extrapolation")
        if bt:
            ax.plot(grid[inside], power_law(grid[inside], bt["params"]), color=C["aqua"], lw=1.4, label="fit on first 70 % (backtest)")
        ax.axhline(pl["l_inf"], color=C["muted"], lw=0.9, ls=":")
        ax.annotate(f"L_inf ≈ {pl['l_inf']:.2f}", (grid[0], pl["l_inf"]), xytext=(4, 4), textcoords="offset points", color=C["ink2"])
        for e in M["loss"]["extrapolation"]:
            ax.scatter([e["step"]], [e["loss"]], s=30, color=C["orange"], edgecolor=C["surface"], linewidth=1, zorder=5)
            ax.annotate(f"{e['loss']:.2f}", (e["step"], e["loss"]), xytext=(0, 9), textcoords="offset points", ha="center", color=C["ink2"])
    ax.set_xscale("log")
    lo, hi = np.quantile(df["loss"][post], [0.002, 0.995])
    ax.set_ylim(min(lo, pl["l_inf"] if pl else lo) - 0.05, hi + 0.05)
    style(ax, "Power-law fit, backtest and extrapolation", "Optimizer step (log)", "Loss")
    legend(ax, loc="upper right")

    ax = axes[1, 1]
    imp = df["loss_improve_per_1k"].where(df["loss_improve_per_1k"] > 0)
    ax.plot(x[post], imp[post], color=C["blue"], label="measured (local quadratic, wide window)")
    if pl:
        deriv = pl["A"] * pl["b"] * x[post] ** (-pl["b"] - 1) * 1000
        ax.plot(x[post], deriv, color=C["orange"], lw=1.8, label="power-law fit")
    ax.set_yscale("log")
    restart_lines(ax, A, label=False)
    style(ax, "Improvement rate", ylabel="Loss gain per 1k steps (log)")
    thousands(ax)
    legend(ax)

    ax = axes[2, 0]
    tok = df["cum_tokens"] / 1e6
    ax.scatter(tok, df["loss"], s=6, color=C["blue"], alpha=0.30, linewidth=0, label="raw")
    ax.plot(tok, df["loss_ema_slow"], color=C["blue_dark"], lw=2, label=f"EMA {A.args.ema_slow}")
    for m in M["loss"]["milestones"]:
        ax.scatter([m["tokens_m"]], [m["loss"]], s=40, color=C["red"], edgecolor=C["surface"], linewidth=1.5, zorder=5)
        ax.annotate(f"{int(m['fraction'] * 100)} %", (m["tokens_m"], m["loss"]), xytext=(6, 6), textcoords="offset points", color=C["ink2"])
    ax.scatter([], [], s=40, color=C["red"], label="share of total improvement")
    ax.set_xscale("log")
    style(ax, "Loss vs tokens processed", "Cumulative tokens (millions, log)", "Loss")
    legend(ax)

    ax = axes[2, 1]
    ax.plot(x, df["ppl"], color=C["blue"])
    ax.set_yscale("log")
    restart_lines(ax, A, label=False)
    style(ax, "Perplexity = exp(EMA loss), if cross-entropy in nats", ylabel="Perplexity (log)")
    thousands(ax)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_optimization(A, path: Path) -> None:
    df, M = A.df, A.M
    x = df["step"].to_numpy()
    n = len(df)
    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5), constrained_layout=True)
    fig.suptitle("Optimization - learning rate and gradients", fontsize=15, fontweight="bold", x=0.01, ha="left")

    ax = axes[0, 0]
    lr = df["learning_rate"].to_numpy()
    ax.plot(x, lr, color=C["aqua"], lw=2.2, label="measured")
    s = M["lr"]["schedule"]
    if "cosine" in s:
        c = s["cosine"]
        t0 = x[A.warm_idx]
        end = max(s["cosine_end_if_min10pct"]["end_step"], x[-1] * 1.2)
        grid = np.linspace(t0, min(end, t0 + c["total_rel"]), 400)
        ax.plot(grid, c["a"] + c["b"] * np.cos(c["k"] * (grid - t0)), color=C["orange"], lw=1.6, ls=(0, (4, 3)), label="cosine fit, extrapolated")
        ax.axvline(x[-1], color=C["muted"], lw=0.9, ls=":")
        ax.annotate("now", (x[-1], lr.max() * 0.35), xytext=(4, 0), textcoords="offset points", color=C["ink2"])
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    style(ax, "Learning-rate schedule", ylabel="LR")
    thousands(ax)
    legend(ax)

    ax = axes[0, 1]
    g = df["grad_norm"].to_numpy()
    ax.scatter(x, g, s=6, color=C["orange"], alpha=0.30, linewidth=0, label="raw")
    ax.plot(x, df["grad_trend"], color="#a4401b", lw=1.8, label="robust trend")
    spike_scatter(ax, x, g, df["grad_spike"].to_numpy() == 1, "grad spike")
    if A.args.grad_clip:
        ax.axhline(A.args.grad_clip, color=C["muted"], lw=1, ls="--", label=f"clip = {A.args.grad_clip:g}")
    ax.set_yscale("log")
    restart_lines(ax, A, label=False)
    style(ax, "Gradient norm and trend", ylabel="Norm (log)")
    thousands(ax)
    legend(ax)

    ax = axes[1, 0]
    post = np.arange(n) >= A.burn
    sc = ax.scatter(df["grad_norm"][post], df["loss"][post], c=x[post], cmap=SEQ_CMAP, s=12, alpha=0.85, linewidth=0)
    ax.set_xscale("log")
    cb = fig.colorbar(sc, ax=ax, pad=0.01)
    cb.set_label("Optimizer step")
    cb.outline.set_visible(False)
    style(ax, "Gradient norm vs loss (colour = step)", "Gradient norm (log)", "Loss")

    ax = axes[1, 1]
    lags, xc, band = df.attrs["xcorr"]
    ax.bar(lags, xc, color=C["blue"], width=0.7)
    ax.axhline(band, color=C["muted"], lw=0.9, ls="--")
    ax.axhline(-band, color=C["muted"], lw=0.9, ls="--", label="95 % band (white noise)")
    ax.axhline(0, color=C["axis"], lw=0.8)
    style(ax, "Cross-correlation: loss residual vs gradient residual", "Lag (measurements; > 0 = gradient lags loss)", "Correlation")
    legend(ax)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_system(A, path: Path) -> None:
    df, M = A.df, A.M
    x = df["step"].to_numpy()
    T = M["timing"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5), constrained_layout=True)
    fig.suptitle("System - throughput, overhead and time budget", fontsize=15, fontweight="bold", x=0.01, ha="left")

    ax = axes[0, 0]
    st = df["tps_steady"]
    ax.scatter(x, st, s=7, color=C["violet"], alpha=0.35, linewidth=0, label="steady intervals")
    ax.plot(x, st.rolling(A.args.window * 2, center=True, min_periods=5).median(), color="#3e30a0", lw=2, label="rolling median")
    ok = st.notna().to_numpy()
    o = ols(x[ok], st[ok])
    ax.plot(x[ok], o["intercept"] + o["slope"] * x[ok], color=C["orange"], lw=1.6, ls=(0, (4, 3)),
            label=f"trend {T['tps_drift_pct_per_1k_steps']:+.3f} %/1k steps")
    restart_lines(ax, A, label=False)
    style(ax, "Steady-state throughput (overhead removed)", ylabel="Tokens / second")
    thousands(ax)
    thousands(ax, "y")
    legend(ax)

    ax = axes[0, 1]
    el = df["elapsed_seconds"].to_numpy()
    bins = np.linspace(el.min(), el.max(), 60)
    for key, color, label in [("steady", C["violet"], "steady"), ("periodic_overhead", C["red"], "periodic overhead"),
                              ("startup", C["muted"], "startup / resume"), ("isolated_slow", C["yellow"], "other slow")]:
        m = (df["interval_kind"] == key).to_numpy()
        if m.any():
            ax.hist(el[m], bins=bins, color=color, alpha=0.9, label=label, edgecolor=C["surface"], linewidth=0.5)
    ax.set_yscale("log")
    style(ax, "Distribution of log-interval durations", "Seconds per interval", "Count (log)")
    legend(ax)

    ax = axes[1, 0]
    ideal = np.arange(1, len(df) + 1) * A.steady_el
    ax.plot(x, df["active_time"] / 60, color=C["blue"], label="active compute")
    ax.plot(x, ideal / 60, color=C["aqua"], lw=1.4, label="ideal (no overhead)")
    if df["wall_time"].notna().all():
        ax.plot(x, df["wall_time"] / 60, color=C["red"], lw=1.4, label="wall clock (incl. crash)")
    restart_lines(ax, A, label=False)
    style(ax, "Cumulative time", ylabel="Minutes")
    thousands(ax)
    legend(ax)

    ax = axes[1, 1]
    b = T["budget"]
    items = [("stable compute", b["compute_stable_s"]), ("periodic overhead", b["periodic_overhead_s"]),
             ("startup", b["startup_overhead_s"]), ("other slow", b["isolated_slow_s"]),
             ("replayed after crash", b["crash_replayed_s"]), ("restart downtime", b["restart_downtime_s"])]
    items = [(k, v) for k, v in items if v > 0][::-1]
    labels = [k for k, _ in items]
    vals = [v for _, v in items]
    ax.barh(labels, vals, color=C["blue"], height=0.6)
    top = max(vals)
    for i, v in enumerate(vals):
        ax.annotate(f"{fmt_dur(v)}  ({100 * v / b['total_s']:.1f} %)", (v, i), xytext=(5, 0), textcoords="offset points", va="center", color=C["ink2"])
    ax.set_xscale("log")
    ax.set_xlim(left=min(vals) * 0.5, right=top * 8)
    ax.grid(axis="y", visible=False)
    style(ax, "Time budget", "Seconds (log)")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_stats(A, path: Path) -> None:
    df, M = A.df, A.M
    S = M["residuals"]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5), constrained_layout=True)
    fig.suptitle("Statistics - residuals, periodicity, correlations, segments", fontsize=15, fontweight="bold", x=0.01, ha="left")
    z = df["loss_z"].to_numpy()[A.burn:]

    ax = axes[0, 0]
    ax.hist(z, bins=50, range=(-6, 6), density=True, color=C["blue"], alpha=0.85, edgecolor=C["surface"], linewidth=0.5, label="standardised residuals")
    grid = np.linspace(-6, 6, 300)
    ax.plot(grid, np.exp(-grid**2 / 2) / math.sqrt(2 * math.pi), color=C["orange"], lw=2, label="N(0, 1)")
    style(ax, f"Residual distribution (kurtosis {S['excess_kurtosis']:+.2f})", "z-score", "Density")
    legend(ax)

    ax = axes[0, 1]
    zs = np.sort(z)
    nd = statistics.NormalDist()
    theo = np.array([nd.inv_cdf((i + 0.5) / len(zs)) for i in range(len(zs))])
    ax.scatter(theo, zs, s=8, color=C["blue"], alpha=0.6, linewidth=0, label="residuals")
    lim = [min(theo.min(), zs.min()), max(theo.max(), zs.max())]
    ax.plot(lim, lim, color=C["orange"], lw=1.6, label="normal reference")
    style(ax, "Q-Q plot vs normal", "Theoretical quantile", "Observed quantile")
    legend(ax)

    ax = axes[0, 2]
    acf = df.attrs["acf"]
    ax.bar(np.arange(len(acf))[1:], acf[1:], color=C["blue"], width=0.7)
    band = 1.96 / math.sqrt(S["n"])
    ax.axhline(band, color=C["muted"], lw=0.9, ls="--")
    ax.axhline(-band, color=C["muted"], lw=0.9, ls="--", label="95 % band")
    ax.axhline(0, color=C["axis"], lw=0.8)
    style(ax, f"Residual autocorrelation (lag 1 = {S['acf_lag1']:+.2f})", "Lag (measurements)", "ACF")
    legend(ax)

    ax = axes[1, 0]
    freqs, power, med, interval = df.attrs["periodogram"]
    keep = freqs > 0
    ax.plot(1 / freqs[keep], power[keep] / med, color=C["blue"], lw=1.2, label="periodogram / median")
    ax.axhline(15, color=C["red"], lw=1.2, ls="--", label="significance (15 × median)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    style(ax, "Periodogram of residuals", "Period (steps, log)", "Power / median (log)")
    legend(ax)

    ax = axes[1, 1]
    corr = df.attrs["corr"]
    im = ax.imshow(corr.values, cmap=DIV_CMAP, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), [c.replace("_res", "") for c in corr.columns])
    ax.set_yticks(range(len(corr)), [c.replace("_res", "") for c in corr.index])
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i, j]:+.2f}", ha="center", va="center", color=C["ink"], fontsize=10)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.outline.set_visible(False)
    style(ax, "Spearman correlation of residuals (steady state)", xlabel=None)

    ax = axes[1, 2]
    seg = A.segments
    pos = np.arange(len(seg))
    gains = seg["loss_gain_per_1k_steps"].to_numpy()
    ax.bar(pos, gains, color=[C["blue"] if g > 0 else C["red"] for g in gains], width=0.65)
    ax.axhline(0, color=C["axis"], lw=0.8)
    ax.set_xticks(pos, seg["segment"])
    ax.set_yscale("symlog", linthresh=max(float(np.abs(gains).min()), 1e-3) if len(gains) else 1e-3)
    style(ax, "Loss gain per 1k steps, by segment (symlog)", "Segment", "Gain per 1k steps")
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- principal
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Analyse approfondie d'un benchmark d'entraînement.")
    p.add_argument("csv", nargs="?", type=Path, default=None, help="CSV du benchmark")
    p.add_argument("--out-dir", type=Path, default=None, help="dossier de sortie (défaut : dossier du CSV)")
    p.add_argument("--window", type=int, default=25, help="fenêtre des statistiques glissantes (mesures)")
    p.add_argument("--trend-window", type=int, default=51, help="fenêtre de la tendance robuste (mesures, impair)")
    p.add_argument("--ema-fast", type=int, default=10, help="span de l'EMA rapide")
    p.add_argument("--ema-slow", type=int, default=100, help="span de l'EMA lente")
    p.add_argument("--z", type=float, default=4.0, help="seuil de détection des pics (écarts-types robustes)")
    p.add_argument("--segments", type=int, default=10, help="nombre de segments pour les statistiques par phase")
    p.add_argument("--grad-clip", type=float, default=None, help="seuil de clipping du gradient (optionnel)")
    p.add_argument("--total-steps", type=int, default=None, help="nombre total de steps prévu (pour l'ETA)")
    return p.parse_args(argv)


def resolve_csv(arg: Path | None) -> Path:
    candidates = [arg] if arg else [DEFAULT_CSV, Path.cwd() / "training_benchmark.csv", Path(__file__).resolve().parent / "training_benchmark.csv"]
    for c in candidates:
        if c is not None and c.exists():
            return c
    raise FileNotFoundError("Benchmark introuvable: " + ", ".join(str(c) for c in candidates if c))


def main(argv=None) -> None:
    args = parse_args(argv)
    csv_path = resolve_csv(args.csv)
    out_dir = args.out_dir or csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    A = analyze(load_raw(csv_path), args)
    apply_style()
    fig_overview(A, out_dir / "benchmark_overview.png")
    fig_loss(A, out_dir / "benchmark_loss.png")
    fig_optimization(A, out_dir / "benchmark_optimization.png")
    fig_system(A, out_dir / "benchmark_system.png")
    fig_stats(A, out_dir / "benchmark_stats.png")

    A.df.to_csv(out_dir / "benchmark_processed.csv", index=False)
    A.segments.to_csv(out_dir / "benchmark_segments.csv", index=False)
    A.anomalies.to_csv(out_dir / "benchmark_anomalies.csv", index=False)
    report = build_report(A, csv_path)
    (out_dir / "benchmark_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (out_dir / "benchmark_report.json").write_text(
        json.dumps(to_jsonable({**A.M, "segments": A.segments}), indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n".join(report))
    print(f"\nFichiers écrits dans: {out_dir}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(f"Erreur: {exc}")
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
import yfinance as yf


@dataclass
class Config:
    assets: tuple = ("AAPL", "MSFT", "AMZN", "GOOGL", "JPM")
    risk_groups: tuple = ("TechnologyAndInternet", "TechnologyAndInternet",
                          "TechnologyAndInternet", "TechnologyAndInternet", "Financials")
    start_date: str = "2019-01-01"
    end_date: str = "2026-01-01"
    momentum_window: int = 20
    volatility_window: int = 20
    factor_names: tuple = ("common", "momentum", "volatility")
    training_window: int = 252
    covariance_window: int = 252
    covariance_half_life: float = 63.0
    specific_half_life: float = 42.0
    refit_every: int = 21
    fit_method: str = "reml"
    profile_penalty: float = 0.10
    parameter_bound: float = 1.0
    variance_floor: float = 1e-6
    variance_ceiling: float = 0.01
    leverage_floor: float = 0.05
    covariance_shrinkage: float = 0.15
    factor_directions: tuple = (0.0, 1.0, 0.0)
    target_volatility: float = 0.10
    gross_limit: float = 1.0
    max_position: float = 0.40
    transaction_cost_bps: float = 10.0
    borrow_rate: float = 0.03
    trading_days: int = 252
    rebalance_every: int = 5
    initial_capital: float = 100000.0
    liquidate_at_end: bool = False


def load_prices(config, data_dir, refresh=False):
    """Load a matching Yahoo snapshot, or download and record its provenance."""
    price_file = data_dir / "yahoo_adjusted_close.csv"
    metadata_file = data_dir / "yahoo_download_metadata.json"
    metadata = {}
    if price_file.exists() and metadata_file.exists() and not refresh:
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        request = metadata.get("request", {})
        covered = (set(config.assets).issubset(request.get("assets", []))
                   and request.get("auto_adjust") is True
                   and request.get("start", "9999") <= config.start_date
                   and request.get("end", "0000") >= config.end_date)
        if covered:
            prices = pd.read_csv(price_file, index_col="date", parse_dates=True,
                                 float_precision="round_trip")
            prices = prices.loc[(prices.index >= config.start_date)
                                & (prices.index < config.end_date), list(config.assets)]
            print("Loaded the saved Yahoo Finance snapshot.")
        else:
            refresh = True
    else:
        refresh = True
    if refresh:
        downloaded = yf.download(
            list(config.assets), start=config.start_date, end=config.end_date,
            interval="1d", auto_adjust=True, keepna=True, threads=False,
            progress=False, timeout=30, group_by="column", multi_level_index=True,
        )
        if downloaded is None or downloaded.empty:
            raise RuntimeError("Yahoo returned no prices. Check the connection and retry.")
        prices = downloaded["Close"].reindex(columns=list(config.assets)).copy()
        prices.index.name = "date"
        data_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "source": "Yahoo Finance via yfinance",
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            "request": {"assets": list(config.assets), "start": config.start_date,
                        "end": config.end_date, "interval": "1d", "auto_adjust": True},
            "actual_first_date": str(prices.index[0].date()),
            "actual_last_date": str(prices.index[-1].date()),
            "rows": len(prices), "yfinance_version": yf.__version__,
        }
        # Validate before saving so a failed request cannot overwrite a good cache.
        validate_prices(prices, config)
        prices.to_csv(price_file, float_format="%.17g")
        metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print("Downloaded real historical prices from Yahoo Finance.")
    prices = prices.astype(float).sort_index()
    prices.index.name = "date"
    prices.columns.name = "asset"
    validate_prices(prices, config)
    return prices, metadata


def validate_prices(prices, config):
    """Require complete prices; never fill missing observations with zeros."""
    if prices.empty or prices.index.has_duplicates:
        raise ValueError("The price table is empty or contains duplicate dates.")
    if not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
        raise ValueError("This implementation requires complete positive prices. Adjust the date range or universe.")
    required = max(config.momentum_window, config.volatility_window) + config.training_window + 10
    if len(prices) < required:
        raise ValueError(f"At least {required} price observations are required.")
    if len(config.assets) <= len(config.factor_names):
        raise ValueError("The number of assets must exceed the number of factors.")
    if len(config.risk_groups) != len(config.assets):
        raise ValueError("Provide exactly one risk group for each asset.")


def cross_section_zscore(values):
    values = np.asarray(values, dtype=float)
    scale = values.std(ddof=0)
    if not np.isfinite(values).all() or scale <= 1e-12:
        raise ValueError("A characteristic has missing values or no cross-sectional variation.")
    return (values - values.mean()) / scale


def build_features(prices, config):
    """Build same-date characteristics using rows through that date only."""
    returns = prices.pct_change(fill_method=None)
    momentum = prices / prices.shift(config.momentum_window) - 1.0
    volatility = returns.rolling(config.volatility_window,
                                 min_periods=config.volatility_window).std(ddof=0)
    exposures, specific_features = {}, {}
    for date in prices.index:
        mom = momentum.loc[date].to_numpy()
        vol = volatility.loc[date].to_numpy()
        if not np.isfinite(mom).all() or not np.isfinite(vol).all():
            continue
        X = np.column_stack([np.ones(len(mom)), cross_section_zscore(mom),
                             cross_section_zscore(vol)])
        if np.linalg.matrix_rank(X) != X.shape[1]:
            raise ValueError(f"Rank-deficient exposure matrix on {date.date()}.")
        exposures[date] = X
        specific_features[date] = cross_section_zscore(np.log(np.maximum(vol, 1e-8)))[:, None]
    return exposures, specific_features


def factor_weights(X, delta):
    """Minimum-specific-variance factor portfolios with W @ X = identity."""
    delta = np.asarray(delta, dtype=float)
    if not np.isfinite(delta).all() or np.any(delta <= 0):
        raise ValueError("Specific variances must be finite and positive.")
    precision = 1.0 / (delta / np.median(delta))
    gram = X.T @ (precision[:, None] * X)
    if np.linalg.cond(gram) > 1e12:
        raise ValueError("The weighted exposure matrix is numerically ill-conditioned.")
    W = np.linalg.solve(gram, X.T * precision[None, :])
    if not np.allclose(W @ X, np.eye(X.shape[1]), atol=1e-9):
        raise ArithmeticError("Factor exposure constraints were not satisfied.")
    return W


def ewma_probabilities(length, half_life):
    ages = np.arange(length - 1, -1, -1, dtype=float)
    weights = np.exp(-np.log(2.0) * ages / half_life)
    return weights / weights.sum()


def project_psd(matrix, floor=1e-10):
    """Symmetrize and floor eigenvalues; report how many were clipped."""
    matrix = (matrix + matrix.T) / 2.0
    eigenvalues, vectors = np.linalg.eigh(matrix)
    count = int(np.sum(eigenvalues < floor))
    projected = (vectors * np.maximum(eigenvalues, floor)) @ vectors.T
    return (projected + projected.T) / 2.0, count


def factor_covariance(history_F, history_A, half_life=63.0, shrinkage=0.15):
    """Separate investable factor-return covariance from latent-factor covariance.

    The plug-in noise correction matches weighted demeaning. It assumes serially
    uncorrelated estimation noise and a locally stable mean. PSD projection and
    fixed diagonal shrinkage are disclosed regularization, not exact recovery.
    """
    F = np.asarray(history_F)
    A = np.asarray(history_A)
    p = ewma_probabilities(len(F), half_life)
    mean = p @ F
    centered = F - mean
    denominator = 1.0 - p @ p
    if denominator <= 0:
        raise ValueError("At least two factor observations are required.")
    empirical_raw = np.einsum("t,ti,tj->ij", p, centered, centered) / denominator
    noise = np.einsum("t,tij->ij", p * (1.0 - p), A) / denominator
    latent_raw = empirical_raw - noise
    latent, clipped = project_psd(latent_raw)
    empirical, _ = project_psd(empirical_raw)
    latent = (1.0 - shrinkage) * latent + shrinkage * np.diag(np.diag(latent))
    empirical = (1.0 - shrinkage) * empirical + shrinkage * np.diag(np.diag(empirical))
    diagnostics = {
        "noise_trace": float(np.trace(noise)),
        "empirical_trace": float(np.trace(empirical_raw)),
        "noise_trace_ratio": float(np.trace(noise) / max(np.trace(empirical_raw), 1e-12)),
        "clipped_factor_eigenvalues": clipped,
    }
    return empirical, latent, mean, diagnostics


def profile_statistics(theta, X, Y, Z, group_index, config):
    """Profile historical factor returns and per-period variance scales."""
    number_groups = int(group_index.max()) + 1
    log_c = np.r_[0.0, theta[:number_groups - 1]]
    gamma = theta[number_groups - 1:]
    log_q = log_c[group_index][None, :] + np.einsum("tnq,q->tn", Z, gamma)
    q = np.exp(log_q)
    inverse_q_X = X / q[:, :, None]
    gram = np.einsum("tni,tnj->tij", X, inverse_q_X)
    rhs = np.einsum("tni,tn->ti", inverse_q_X, Y)
    F = np.linalg.solve(gram, rhs[:, :, None])[:, :, 0]
    residual = Y - np.einsum("tnk,tk->tn", X, F)
    sse = np.sum(residual ** 2 / q, axis=1)
    n, k = X.shape[1:]
    dof = n - k if config.fit_method == "reml" else n
    likelihood_scale = np.clip(sse / dof, config.variance_floor, config.variance_ceiling)
    terms = dof * np.log(2.0 * np.pi * likelihood_scale) + log_q.sum(axis=1) + sse / likelihood_scale
    if config.fit_method == "reml":
        sign, logdet = np.linalg.slogdet(gram)
        if np.any(sign <= 0):
            raise ValueError("Non-positive restricted-likelihood information matrix.")
        terms = terms + logdet
    objective = 0.5 * terms.mean() + 0.5 * config.profile_penalty * np.sum(theta ** 2)
    # Explicit n-k correction for risk forecasts, including when fitting by MLE.
    risk_scale = np.clip(sse / (n - k), config.variance_floor, config.variance_ceiling)
    return objective, risk_scale, likelihood_scale


def fit_structural_model(theta, X, Y, Z, group_index, config):
    """Bounded penalized profile likelihood, with explicit convergence fallback."""
    objective = lambda value: profile_statistics(value, X, Y, Z, group_index, config)[0]
    result = minimize(objective, theta, method="L-BFGS-B",
                      bounds=[(-config.parameter_bound, config.parameter_bound)] * len(theta),
                      options={"maxiter": 150, "ftol": 1e-9})
    accepted = bool(result.success and np.isfinite(result.fun))
    chosen = np.asarray(result.x if accepted else theta).copy()
    diagnostics = {
        "optimizer_success": accepted, "optimizer_status": int(result.status),
        "optimizer_message": str(result.message), "iterations": int(result.nit),
        "objective": float(objective(chosen)),
        "boundary_hits": int(np.sum(np.abs(chosen) >= config.parameter_bound - 1e-4)),
        "fallback_to_previous_parameters": not accepted,
    }
    return chosen, diagnostics


def allocation_weights(W, empirical_cov, asset_cov, config):
    """Map disclosed factor directions to stocks, then apply uniform risk/cap scaling.

    Inverse-volatility allocation is not exact equal-risk contribution when
    factors are correlated. Directions are assumptions, not optimized forecasts.
    """
    directions = np.asarray(config.factor_directions, dtype=float)
    a = directions / np.sqrt(np.maximum(np.diag(empirical_cov), 1e-10))
    if not np.any(np.abs(a) > 0):
        return np.zeros(W.shape[1]), {"predicted_volatility": 0.0, "scale": 0.0}
    daily_target = config.target_volatility / np.sqrt(config.trading_days)
    a *= daily_target / np.sqrt(max(a @ empirical_cov @ a, 1e-12))
    raw = W.T @ a
    raw_risk = np.sqrt(max(raw @ asset_cov @ raw, 1e-12))
    scale = min(daily_target / raw_risk,
                config.gross_limit / max(np.abs(raw).sum(), 1e-12),
                config.max_position / max(np.abs(raw).max(), 1e-12))
    w = raw * scale
    return w, {"predicted_volatility": float(np.sqrt(w @ asset_cov @ w) * np.sqrt(config.trading_days)),
               "scale": float(scale), "gross_exposure": float(np.abs(w).sum()),
               "net_exposure": float(w.sum())}


def run_model(prices, exposures, specific_features, config, method):
    """Generate pre-return risk forecasts and never rewrite past predictions."""
    dates = prices.index
    returns = prices.pct_change(fill_method=None)
    group_names = list(dict.fromkeys(config.risk_groups))
    group_index = np.array([group_names.index(value) for value in config.risk_groups])
    theta = np.zeros(len(group_names) - 1 + 1)
    alpha = 1.0 - np.exp(-np.log(2.0) / config.specific_half_life)
    factor_history, noise_history = [], []
    train_X, train_Y, train_Z = [], [], []
    F_estimates, residual_estimates, forecasts = {}, {}, {}
    weights_history, targets = {}, {}
    risk_rows, fit_rows, parameter_rows = [], [], []
    covariance_rows, latent_rows, empirical_rows = [], [], []
    factor_weight_rows, specific_rows = [], []
    last_fit_count = -config.refit_every
    state, previous = None, None
    for date_index, date in enumerate(dates):
        if date not in exposures:
            continue
        X, Z = exposures[date], specific_features[date]
        if state is None:
            state = returns.iloc[max(1, date_index - config.volatility_window + 1):date_index + 1].var(ddof=0).to_numpy()
            state = np.clip(state, config.variance_floor, config.variance_ceiling)
        if previous is not None:
            prior_date = previous["date"]
            realized = returns.loc[date].to_numpy()
            F = previous["W"] @ realized
            residual = realized - previous["X"] @ F
            leverage = np.einsum("ij,ji->i", previous["X"], previous["W"])
            adjusted_square = residual ** 2 / np.maximum(1.0 - leverage, config.leverage_floor)
            state = np.clip((1.0 - alpha) * state + alpha * adjusted_square,
                            config.variance_floor, config.variance_ceiling)
            factor_history.append(F)
            noise_history.append(previous["A"])
            F_estimates[prior_date] = F
            residual_estimates[prior_date] = residual
            train_X.append(previous["X"])
            train_Y.append(realized)
            train_Z.append(previous["Z"])
            if previous.get("eligible"):
                error = realized - previous["mean"]
                covariance = previous["asset_cov"]
                quadratic = float(error @ np.linalg.solve(covariance, error))
                risk_rows.append({"signal_date": prior_date, "return_end_date": date,
                                  "mahalanobis_per_asset": quadratic / len(realized),
                                  "gaussian_nll_per_asset": 0.5 * (np.linalg.slogdet(covariance)[1]
                                      + quadratic + len(realized) * np.log(2.0 * np.pi)) / len(realized),
                                  "max_exposure_error": previous["exposure_error"],
                                  "clipped_factor_eigenvalues": previous["cov_diagnostics"]["clipped_factor_eigenvalues"],
                                  "noise_trace_ratio": previous["cov_diagnostics"]["noise_trace_ratio"]})
                risk_rows[-1]["max_residual_leverage"] = float(leverage.max())
                risk_rows[-1]["leverage_floor_activations"] = int(np.sum(1.0 - leverage < config.leverage_floor))
        eligible = len(train_Y) >= config.training_window
        specific = state.copy()
        if method == "structural" and eligible:
            X_history = np.stack(train_X[-config.training_window:])
            Y_history = np.stack(train_Y[-config.training_window:])
            Z_history = np.stack(train_Z[-config.training_window:])
            if len(train_Y) - last_fit_count >= config.refit_every:
                theta, fit = fit_structural_model(theta, X_history, Y_history, Z_history, group_index, config)
                fit_rows.append({"date": date, **fit})
                last_fit_count = len(train_Y)
            _, scales, likelihood_scales = profile_statistics(theta, X_history, Y_history,
                                                             Z_history, group_index, config)
            scale_weights = ewma_probabilities(len(scales), config.specific_half_life)
            predicted_scale = float(scale_weights @ scales)
            log_c = np.r_[0.0, theta[:len(group_names) - 1]]
            gamma = theta[len(group_names) - 1:]
            q_current = np.exp(log_c[group_index] + Z @ gamma)
            specific = np.clip(predicted_scale * q_current, config.variance_floor, config.variance_ceiling)
            parameters = {f"c_{name}": float(np.exp(log_c[j])) for j, name in enumerate(group_names)}
            parameter_rows.append({"date": date, "predicted_scale": predicted_scale,
                                   "last_likelihood_scale": float(likelihood_scales[-1]),
                                   "gamma_log_volatility": float(gamma[0]), **parameters})
        W = factor_weights(X, specific)
        A = (W * specific[None, :]) @ W.T
        weights_history[date] = W
        cov_diagnostics = {"clipped_factor_eigenvalues": 0, "noise_trace_ratio": 0.0}
        if len(factor_history) >= 2:
            empirical, latent, factor_mean, cov_diagnostics = factor_covariance(
                factor_history[-config.covariance_window:], noise_history[-config.covariance_window:],
                config.covariance_half_life, config.covariance_shrinkage)
        else:
            empirical = latent = np.eye(X.shape[1]) * 1e-4
            factor_mean = np.zeros(X.shape[1])
        asset_cov = X @ latent @ X.T + np.diag(specific)
        exposure_error = float(np.max(np.abs(W @ X - np.eye(X.shape[1]))))
        forecasts[date] = {"delta": specific, "asset_cov": asset_cov,
                           "factor_empirical": empirical, "factor_latent": latent,
                           "mean": X @ factor_mean}
        if eligible:
            target, allocation = allocation_weights(W, empirical, asset_cov, config)
            targets[date] = target
            specific_rows.append({"date": date, **dict(zip(config.assets, specific))})
            for factor_index, factor in enumerate(config.factor_names):
                factor_weight_rows.append({"date": date, "factor": factor,
                                          **dict(zip(config.assets, W[factor_index]))})
            covariance_rows.append({"date": date, **{f"{a}__{b}": float(asset_cov[i, j])
                                   for i, a in enumerate(config.assets) for j, b in enumerate(config.assets)}})
            latent_rows.append({"date": date, **{f"{a}__{b}": float(latent[i, j])
                               for i, a in enumerate(config.factor_names) for j, b in enumerate(config.factor_names)}})
            empirical_rows.append({"date": date, **{f"{a}__{b}": float(empirical[i, j])
                                  for i, a in enumerate(config.factor_names) for j, b in enumerate(config.factor_names)}})
            forecasts[date]["allocation"] = allocation
            forecasts[date]["allocation"]["specific_variance_floor_hits"] = int(np.sum(specific <= config.variance_floor))
            forecasts[date]["allocation"]["specific_variance_ceiling_hits"] = int(np.sum(specific >= config.variance_ceiling))
        previous = {"date": date, "X": X, "Z": Z, "W": W, "A": A,
                    "eligible": eligible, "asset_cov": asset_cov, "mean": X @ factor_mean,
                    "exposure_error": exposure_error, "cov_diagnostics": cov_diagnostics}
        if eligible and (len(train_Y) - config.training_window) % 250 == 0:
            print(f"{method}: processed {date.date()} ({len(train_Y)} completed model periods)", flush=True)
    if not targets:
        raise ValueError("No out-of-sample targets were generated.")
    target_table = pd.DataFrame.from_dict(targets, orient="index", columns=config.assets).sort_index()
    target_table.index.name = "signal_date"
    factor_table = pd.DataFrame.from_dict(F_estimates, orient="index", columns=config.factor_names).sort_index()
    factor_table.index.name = "period_start_date"
    return {
        "targets": target_table, "factor_returns": factor_table,
        "factor_estimates": F_estimates, "residual_estimates": residual_estimates,
        "forecasts": forecasts, "weights": weights_history,
        "risk_evaluation": pd.DataFrame(risk_rows),
        "fit_diagnostics": pd.DataFrame(fit_rows, columns=[
            "date", "optimizer_success", "optimizer_status", "optimizer_message",
            "iterations", "objective", "boundary_hits", "fallback_to_previous_parameters"]),
        "structural_parameters": pd.DataFrame(parameter_rows, columns=[
            "date", "predicted_scale", "last_likelihood_scale", "gamma_log_volatility",
            *[f"c_{name}" for name in group_names]]),
        "factor_weights": pd.DataFrame(factor_weight_rows),
        "specific_variance_forecasts": pd.DataFrame(specific_rows),
        "asset_covariance_forecasts": pd.DataFrame(covariance_rows),
        "latent_factor_covariance": pd.DataFrame(latent_rows),
        "empirical_factor_covariance": pd.DataFrame(empirical_rows),
    }


# PORTFOLIO_AND_TEST_HELPERS

def execution_backtest(prices, targets, exposures, factor_estimates,
                       residual_estimates, config):
    """Execute signal s at close s+1; earn its first return from s+1 to s+2.

    Holdings and cash are dollar values. Targets are post-cost NAV weights.
    Turnover is full L1 traded notional divided by pre-trade NAV. Fee columns
    in daily are positive dollars; attribution fee columns are negative dollars.
    The ledger begins at the first execution close, including its entry fee.
    Final holdings remain open unless config.liquidate_at_end is explicitly true.
    """
    prices, targets = prices.copy(), targets.copy()
    for label, frame in (("prices", prices), ("targets", targets)):
        if frame.empty or not frame.index.is_monotonic_increasing:
            raise ValueError(label + " must be nonempty and chronologically sorted")
        if not frame.index.is_unique or not frame.columns.is_unique:
            raise ValueError(label + " must have unique dates and assets")
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError(label + " contains missing or nonfinite values")
    if (prices <= 0).any().any() or set(targets.columns) != set(prices.columns):
        raise ValueError("Require positive adjusted closes and matching target assets")
    targets = targets.reindex(columns=prices.columns)
    if not targets.index.isin(prices.index).all():
        raise ValueError("Each signal date must be a price date")
    tc = float(getattr(config, "transaction_cost_bps", 10)) / 10000.0
    borrow = float(getattr(config, "borrow_rate", .03))
    days = int(getattr(config, "trading_days", 252))
    every = int(getattr(config, "rebalance_every", 5))
    capital = float(getattr(config, "initial_capital", 100000))
    factors = list(config.factor_names)
    if not (0 <= tc < 1 and borrow >= 0 and days > 0 and every > 0 and capital > 0):
        raise ValueError("Invalid costs, annualization, rebalance interval, or capital")
    if len(set(factors + ["specific", "transaction_cost", "borrow_cost"])) != len(factors) + 3:
        raise ValueError("Factor names must be unique and not reserved accounting names")
    schedule, pending = {}, []
    for count, (signal, row) in enumerate(targets.iterrows()):
        if count % every:
            continue
        pos = prices.index.get_loc(signal) + 1
        if pos == len(prices):
            pending.append({"signal_date": signal, "reason": "No next trading close"})
        else:
            schedule[prices.index[pos]] = (signal, row.to_numpy(dtype=float))
    if not schedule:
        raise ValueError("No signal has a next-close execution date")
    first = prices.index.get_loc(min(schedule))
    if first >= len(prices) - 1:
        raise ValueError("Need at least one return interval after first execution")
    returns = prices.pct_change(fill_method=None)
    values, cash, nav = np.zeros(prices.shape[1]), capital, capital
    ledger, weights, attribution, trades = [], [], [], []
    liquidate = bool(getattr(config, "liquidate_at_end", False))
    for pos in range(first, len(prices)):
        date, prev_nav = prices.index[pos], nav
        asset_r = returns.iloc[pos].to_numpy(dtype=float)
        asset_pnl = values * asset_r
        gross_pnl = float(asset_pnl.sum())
        borrow_cost = float(-np.minimum(values, 0).sum() * borrow / days)
        components = np.zeros(len(factors) + 3)
        if np.any(values):
            previous = prices.index[pos - 1]
            if any(previous not in series for series in
                   (exposures, factor_estimates, residual_estimates)):
                raise ValueError("Missing realized attribution inputs for " + str(previous))
            x = np.asarray(exposures[previous], dtype=float)
            f = np.asarray(factor_estimates[previous], dtype=float).reshape(-1)
            eps = np.asarray(residual_estimates[previous], dtype=float).reshape(-1)
            if x.shape != (len(values), len(factors)) or f.shape != (len(factors),):
                raise ValueError("Attribution exposure/factor dimensions disagree")
            if eps.shape != values.shape or not np.allclose(x @ f + eps, asset_r,
                                                          atol=1e-9, rtol=1e-6):
                raise ValueError("Factor/residual estimates do not reconstruct realized returns")
            components[:len(factors)] = (values @ x) * f
            components[len(factors)] = float(values @ eps)
        drifted = values + asset_pnl
        cash -= borrow_cost
        pretrade = float(cash + drifted.sum())
        if not np.isfinite(pretrade) or pretrade <= 0:
            raise ValueError("Portfolio insolvent before trading on " + str(date))
        transaction_cost, turnover = 0.0, 0.0
        instruction = schedule.get(date)
        if liquidate and pos == len(prices) - 1:
            instruction = (pd.NaT, np.zeros(len(values)))
        if instruction is not None:
            signal, target = instruction
            if tc * np.abs(target).sum() >= 1:
                raise ValueError("Target leverage prevents unique self-financing cost solution")
            def balance(equity):
                return equity + tc * np.abs(target * equity - drifted).sum() - pretrade
            if balance(0.0) >= 0:
                raise ValueError("Cannot fund liquidation/rebalance costs on " + str(date))
            nav = brentq(balance, 0.0, pretrade, xtol=1e-10)
            values = target * nav
            change = values - drifted
            traded = float(np.abs(change).sum())
            transaction_cost, turnover = tc * traded, traded / pretrade
            cash = float(nav - values.sum())
            for asset, notional in zip(prices.columns, change):
                if abs(notional) > 1e-9:
                    trades.append({"signal_date": signal, "execution_date": date,
                                   "asset": asset, "notional_change": float(notional)})
        else:
            values, nav = drifted, pretrade
        if nav <= 0 or not np.isfinite(nav):
            raise ValueError("Portfolio insolvent after trading on " + str(date))
        components[-2:] = (-transaction_cost, -borrow_cost)
        if not np.isclose(components.sum(), nav - prev_nav, atol=1e-7, rtol=1e-8):
            raise ArithmeticError("Dollar attribution failed to reconcile on " + str(date))
        ledger.append({"date": date, "prev_nav": prev_nav, "nav": nav,
                       "gross_return": gross_pnl / prev_nav,
                       "net_return": nav / prev_nav - 1,
                       "transaction_cost": transaction_cost, "borrow_cost": borrow_cost,
                       "turnover": turnover, "gross_exposure": np.abs(values).sum() / nav,
                       "net_exposure": values.sum() / nav, "cash": cash})
        weights.append(values / nav)
        attribution.append(components)
    daily = pd.DataFrame(ledger).set_index("date")
    return {"daily": daily,
            "held_weights": pd.DataFrame(weights, index=daily.index, columns=prices.columns),
            "attribution": pd.DataFrame(attribution, index=daily.index,
                                        columns=factors + ["specific", "transaction_cost", "borrow_cost"]),
            "trades": pd.DataFrame(trades, columns=["signal_date", "execution_date", "asset", "notional_change"]),
            "pending_signals": pd.DataFrame(pending, columns=["signal_date", "reason"]),
            "terminal_liquidated": liquidate}


def performance_summary(daily, trading_days=252):
    """Summarize actual ledger NAV; include entry costs without a zero warmup.

    First ledger row is instantaneous entry. Combine that entry fee with the
    first earned interval for daily-return statistics, using len(daily)-1 days.
    Annual turnover includes entry and any explicitly requested terminal trades.
    """
    if len(daily) < 2 or (daily.nav <= 0).any() or trading_days <= 0:
        raise ValueError("Require positive NAV and at least one earned return interval")
    initial = float(daily.prev_nav.iloc[0])
    periods = len(daily) - 1
    r = daily.net_return.iloc[1:].copy()
    r.iloc[0] = daily.nav.iloc[1] / initial - 1
    std = float(r.std(ddof=1)) if len(r) > 1 else float("nan")
    wealth = np.r_[initial, daily.nav.to_numpy()]
    drawdown = wealth / np.maximum.accumulate(wealth) - 1
    total = float(daily.nav.iloc[-1] / initial - 1)
    annual_return = float(np.expm1(np.log1p(total) * trading_days / periods))
    return {"annual_return": annual_return, "annual_vol": std * np.sqrt(trading_days),
            "sharpe": float(r.mean() / std * np.sqrt(trading_days)) if std > 0 else float("nan"),
            "max_drawdown": float(drawdown.min()), "total_return": total,
            "mean_gross": float(daily.gross_exposure.iloc[:-1].mean()),
            "annualized_turnover": float(daily.turnover.sum() * trading_days / periods),
            "total_fees": float(daily.transaction_cost.sum() + daily.borrow_cost.sum())}


def make_project_plots(results, factor_returns, latest_X, output_dir):
    """Save three English PNGs; unit-factor sums are PnL, not compounded wealth."""
    if not results:
        raise ValueError("Need at least one backtest result")
    comparison_dates = next(iter(results.values()))["daily"].index
    if any(not result["daily"].index.equals(comparison_dates) for result in results.values()):
        raise ValueError("All strategies must share identical execution/valuation dates")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 120, "font.size": 10, "axes.grid": True,
                         "grid.alpha": .2, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for name, result in results.items():
        d = result["daily"]
        initial = float(d.prev_nav.iloc[0])
        nav = d.nav / initial
        high = np.maximum.accumulate(np.r_[1., nav.to_numpy()])[1:]
        axes[0, 0].plot(d.index, nav, label=name)
        axes[0, 1].plot(d.index, 100 * (nav / high - 1), label=name)
        axes[1, 0].plot(d.index, 100 * (nav - 1), label=name + " net")
        axes[1, 0].plot(d.index, 100 * ((1 + d.gross_return).cumprod() - 1),
                        linestyle="--", alpha=.7, label=name + " gross-return index")
        axes[1, 1].plot(d.index, (d.transaction_cost + d.borrow_cost).cumsum(), label=name)
    titles = [("Net NAV / initial capital", "NAV index"), ("Drawdown from initial/running peak", "%"),
              ("Gross-return index and net NAV return", "%"), ("Cumulative fees", "Dollars")]
    for ax, (title, ylabel) in zip(axes.flat, titles):
        ax.set(title=title, ylabel=ylabel)
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Historical sample comparison: identical execution dates")
    performance_path = output_dir / "performance.png"
    fig.savefig(performance_path)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    factor_returns.cumsum().plot(ax=axes[0])
    factor_returns.rolling(63, min_periods=2).std().mul(np.sqrt(252)).plot(ax=axes[1])
    axes[0].set(title="Cumulative unit-factor PnL (sum of realized factor returns)", ylabel="PnL per exposure unit")
    axes[1].set(title="63-day rolling annualized factor-return volatility", ylabel="Volatility per exposure unit")
    fig.suptitle("Estimated factor portfolios: exposure units are not cash-investment weights")
    factor_path = output_dir / "factor_diagnostics.png"
    fig.savefig(factor_path)
    plt.close(fig)
    latest = pd.DataFrame({name: result["held_weights"].iloc[-1]
                           for name, result in results.items()})
    x = latest_X.reindex(index=latest.index)
    if x.isna().any().any():
        raise ValueError("latest_X must cover every latest held asset")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    latest.plot.bar(ax=axes[0])
    pd.DataFrame(x.to_numpy().T @ latest.to_numpy(), index=x.columns,
                 columns=latest.columns).plot.bar(ax=axes[1])
    axes[0].set(title="Latest marked holdings", ylabel="Weight / NAV")
    axes[1].set(title="Latest holdings multiplied by latest X", ylabel="Factor exposure")
    for ax in axes:
        ax.axhline(0, color="black", linewidth=.7)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Small historical illustration: " + str(len(latest)) + " stocks; no universe-selection claim")
    weights_path = output_dir / "latest_weights.png"
    fig.savefig(weights_path)
    plt.close(fig)
    return {"performance": performance_path, "factor_diagnostics": factor_path,
            "latest_weights": weights_path}


def run_self_tests(namespace=None):
    """Check mathematical identities, risk forecasts, timing, and ledger cash flows."""
    import dataclasses
    from types import SimpleNamespace
    import numpy as np
    import pandas as pd

    ns = globals() if namespace is None else namespace
    factor_weights = ns["factor_weights"]
    factor_covariance = ns["factor_covariance"]
    project_psd = ns["project_psd"]
    allocation_weights = ns["allocation_weights"]
    build_features = ns["build_features"]
    execution_backtest = ns["execution_backtest"]
    cfg = dataclasses.replace(ns["Config"](), factor_directions=(0, 1, 0),
                              target_volatility=0.1, trading_days=252,
                              gross_limit=1.0, max_position=0.4,
                              momentum_window=20, volatility_window=20)
    checked = []
    close = np.testing.assert_allclose
    rng = np.random.default_rng(1841)
    scores = rng.normal(size=(5, 2))
    scores = (scores - scores.mean(axis=0)) / scores.std(axis=0)
    X = np.column_stack([np.ones(5), scores])
    delta = 1e-4 * np.array([0.7, 1.1, 1.8, 0.9, 1.4])
    Delta = np.diag(delta)
    SigmaF = 1e-4 * np.array([[1.0, 0.2, -0.1], [0.2, 0.8, 0.15],
                             [-0.1, 0.15, 0.6]])
    W = factor_weights(X, delta)
    close(W @ X, np.eye(3), atol=1e-11)
    A = W @ Delta @ W.T
    close(A, np.linalg.inv(X.T @ (X / delta[:, None])), atol=1e-12)
    asset_cov = X @ SigmaF @ X.T + Delta
    close(W @ asset_cov @ W.T, SigmaF + A, atol=1e-12)
    nullspace = np.linalg.svd(X.T, full_matrices=True)[2][3:].T
    for unused in range(10):
        alternative = W + rng.normal(size=(3, 2)) @ nullspace.T
        close(alternative @ X, np.eye(3), atol=1e-11)
        difference = alternative @ asset_cov @ alternative.T - W @ asset_cov @ W.T
        assert np.linalg.eigvalsh(difference).min() >= -1e-12
    checked.append("unit exposures and minimum variance among feasible portfolios")

    draws = 50000
    factors = rng.multivariate_normal(np.zeros(3), SigmaF, size=draws)
    errors = rng.normal(size=(draws, 5)) * np.sqrt(delta)
    returns = factors @ X.T + errors
    estimated = returns @ W.T
    empirical, latent, mean, diagnostics = factor_covariance(
        estimated, np.broadcast_to(A, (draws, 3, 3)),
        half_life=1e12, shrinkage=0.0)
    assert np.linalg.norm(empirical - (SigmaF + A)) / np.linalg.norm(SigmaF + A) < 0.04
    assert np.linalg.norm(latent - SigmaF) / np.linalg.norm(SigmaF) < 0.04
    assert np.linalg.norm(np.cov(returns, rowvar=False) - asset_cov) / np.linalg.norm(asset_cov) < 0.04
    close(W @ (X @ latent @ X.T + Delta) @ W.T, empirical, atol=1e-10)
    bad_asset_cov = X @ empirical @ X.T + Delta
    close(W @ bad_asset_cov @ W.T - empirical, A, atol=1e-10)
    residuals = returns - estimated @ X.T
    leverage = np.sum(X * W.T, axis=1)
    close(np.mean(residuals ** 2, axis=0) / delta, 1.0 - leverage, rtol=0.03)
    close(np.mean(residuals ** 2 / (1.0 - leverage), axis=0) / delta,
          np.ones(5), rtol=0.03)
    checked.append("Monte Carlo covariance correction and residual leverage correction")

    profile = ns["profile_statistics"]
    profile_draws = 20000
    profile_X = np.broadcast_to(X, (profile_draws, 5, 3))
    profile_Y = factors[:profile_draws] @ X.T + rng.normal(scale=0.01, size=(profile_draws, 5))
    profile_Z = np.zeros((profile_draws, 5, 1))
    profile_groups = np.zeros(5, dtype=int)
    reml_cfg = dataclasses.replace(cfg, fit_method="reml", variance_floor=1e-12,
                                   variance_ceiling=1.0, profile_penalty=0.0)
    objective, reml_risk, reml_scale = profile(np.zeros(1), profile_X, profile_Y,
                                              profile_Z, profile_groups, reml_cfg)
    assert np.isfinite(objective)
    assert abs(reml_scale.mean() / 1e-4 - 1.0) < 0.04
    mle_cfg = dataclasses.replace(reml_cfg, fit_method="mle")
    unused, mle_risk, mle_scale = profile(np.zeros(1), profile_X, profile_Y,
                                          profile_Z, profile_groups, mle_cfg)
    close(mle_scale, reml_scale * (5 - 3) / 5, atol=1e-12)
    close(mle_risk, reml_risk, atol=1e-12)
    checked.append("profile MLE bias, REML scale, and explicit risk degrees-of-freedom correction")

    short_F = rng.normal(scale=0.03, size=(12, 3))
    short_A = np.array([np.eye(3) * (i + 1) * 1e-6 for i in range(12)])
    p = 2.0 ** (-np.arange(11, -1, -1) / 2.0)
    p /= p.sum()
    centered = short_F - p @ short_F
    expected_S = (centered.T * p) @ centered / (1.0 - p @ p)
    expected_B = np.einsum("s,sij->ij", p * (1.0 - p), short_A) / (1.0 - p @ p)
    short_emp, short_latent, short_mean, unused = factor_covariance(
        short_F, short_A, half_life=2.0, shrinkage=0.0)
    close(short_emp, expected_S, atol=1e-12)
    close(short_mean, p @ short_F, atol=1e-12)
    expected_latent, unused = project_psd(expected_S - expected_B)
    close(short_latent, expected_latent, atol=1e-10)
    projected, clipped = project_psd(np.diag([-2.0, 0.5, 1.0]), floor=1e-10)
    close(projected, np.diag([1e-10, 0.5, 1.0]), atol=1e-12)
    assert clipped == 1
    checked.append("finite-weight noise subtraction and positive covariance projection")

    w, allocation = allocation_weights(W, empirical, asset_cov, cfg)
    exposures = X.T @ w
    assert exposures[1] > 0
    close(exposures[[0, 2]], np.zeros(2), atol=1e-10)
    assert np.abs(w).sum() <= cfg.gross_limit + 1e-10
    assert np.abs(w).max() <= cfg.max_position + 1e-10
    assert np.sqrt(w @ asset_cov @ w * cfg.trading_days) <= cfg.target_volatility + 1e-10
    close(w @ asset_cov @ w, exposures @ (W @ asset_cov @ W.T) @ exposures,
          atol=1e-12)
    checked.append("allocation direction, neutrality, exposure caps, and variance consistency")

    dates = pd.bdate_range("2020-01-01", periods=100)
    prices = pd.DataFrame(100.0 * np.exp(np.cumsum(
        rng.normal(scale=0.01, size=(100, 5)), axis=0)),
        index=dates, columns=["A", "B", "C", "D", "E"])
    original_X, original_Z = build_features(prices, cfg)
    changed = prices.copy()
    cutoff = dates[70]
    changed.loc[changed.index > cutoff] *= np.exp(
        rng.normal(scale=0.4, size=changed.loc[changed.index > cutoff].shape))
    changed_X, changed_Z = build_features(changed, cfg)
    early = [date for date in original_X if date <= cutoff]
    assert early and set(early) == {date for date in changed_X if date <= cutoff}
    for date in early:
        close(original_X[date], changed_X[date], atol=1e-12)
        close(original_Z[date], changed_Z[date], atol=1e-12)
    checked.append("feature construction does not use future price rows")

    six = pd.bdate_range("2020-01-01", periods=6)
    ledger_prices = pd.DataFrame({"A": [100, 100, 110, 110, 121, 121],
                                  "B": [100, 100, 90, 90, 81, 81]}, index=six)
    ledger_cfg = dict(vars(cfg))
    ledger_cfg.update(transaction_cost_bps=10.0, borrow_rate=0.03,
                      rebalance_every=1, initial_capital=100000.0,
                      factor_names=("common",), liquidate_at_end=False)
    ledger_cfg = SimpleNamespace(**ledger_cfg)
    ledger_X = {date: np.ones((2, 1)) for date in six}
    ledger_F = {date: np.zeros(1) for date in six}
    ledger_e = {six[i]: ledger_prices.iloc[i + 1].to_numpy(dtype=float) /
                ledger_prices.iloc[i].to_numpy(dtype=float) - 1.0 for i in range(5)}
    targets = pd.DataFrame([[0.5, -0.5]], index=six[:1], columns=["A", "B"])
    result = execution_backtest(ledger_prices, targets,
                               ledger_X, ledger_F, ledger_e, ledger_cfg)
    daily, attribution = result["daily"], result["attribution"]
    entry_nav = ledger_cfg.initial_capital / 1.001
    close(daily.loc[six[1], "nav"], entry_nav, rtol=1e-11)
    close(daily.loc[six[1], "gross_return"], 0.0, atol=1e-12)
    close(daily.loc[six[1], "transaction_cost"], 100000.0 - entry_nav, rtol=1e-10)
    borrow = 0.5 * entry_nav * ledger_cfg.borrow_rate / ledger_cfg.trading_days
    close(daily.loc[six[2], "gross_return"], 0.1, atol=1e-12)
    close(daily.loc[six[2], "borrow_cost"], borrow, rtol=1e-11)
    close(daily.loc[six[2], "nav"], entry_nav * 1.1 - borrow, rtol=1e-11)
    close(attribution.sum(axis=1).to_numpy(),
          (daily["nav"] - daily["prev_nav"]).to_numpy(), atol=1e-8)
    assert set(pd.to_datetime(result["trades"]["execution_date"])) == {six[1]}
    assert not result["terminal_liquidated"]
    checked.append("next-close execution, entry fees, short borrow, and cash-flow attribution")
    print("Offline self-tests passed (%d groups)." % len(checked))
    return checked

# END_PORTFOLIO_AND_TEST_HELPERS


def save_model_outputs(result, destination):
    destination.mkdir(parents=True, exist_ok=True)
    table_names = ("targets", "factor_returns", "risk_evaluation", "fit_diagnostics",
                   "structural_parameters", "factor_weights", "specific_variance_forecasts",
                   "asset_covariance_forecasts", "latent_factor_covariance", "empirical_factor_covariance")
    for name in table_names:
        result[name].to_csv(destination / f"{name}.csv", index=name in ("targets", "factor_returns"))
    for name in ("daily", "held_weights", "attribution", "trades", "pending_signals"):
        if name in result:
            result[name].to_csv(destination / f"{name}.csv", index=True)
    latest_date = result["targets"].index[-1]
    forecast = result["forecasts"][latest_date]
    latest = {"signal_date": str(latest_date.date()),
              "target_weights": result["targets"].iloc[-1].to_dict(),
              "allocation": forecast["allocation"]}
    (destination / "latest_signal.json").write_text(json.dumps(latest, indent=2), encoding="utf-8")


def main(argv=None):
    defaults = Config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument("--start", default=defaults.start_date)
    parser.add_argument("--end", default=defaults.end_date)
    parser.add_argument("--fit-method", choices=("mle", "reml"), default=defaults.fit_method)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--factor-directions", default=",".join(map(str, defaults.factor_directions)))
    parser.add_argument("--liquidate-at-end", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    config = Config(start_date=args.start, end_date=args.end, fit_method=args.fit_method,
                    factor_directions=tuple(float(value) for value in args.factor_directions.split(",")),
                    liquidate_at_end=args.liquidate_at_end)
    if len(config.factor_directions) != len(config.factor_names):
        parser.error("Provide three factor directions: common,momentum,volatility.")
    if args.self_test:
        run_self_tests()
        return
    base = Path(__file__).resolve().parent
    output = args.output_dir or base / "results"
    output.mkdir(parents=True, exist_ok=True)
    prices, metadata = load_prices(config, base / "data", refresh=args.refresh_data)
    print(f"Real Yahoo data: {len(prices)} rows, {prices.index[0].date()} to {prices.index[-1].date()}.")
    exposures, specific_features = build_features(prices, config)
    latest_date = max(exposures)
    latest_X = pd.DataFrame(exposures[latest_date], index=config.assets, columns=config.factor_names)
    latest_X.to_csv(output / "latest_exposures.csv", index_label="asset")
    feature_rows = [{"date": date, "asset": asset, **dict(zip(config.factor_names, X[i])),
                     "specific_log_volatility": float(specific_features[date][i, 0])}
                    for date, X in exposures.items() for i, asset in enumerate(config.assets)]
    pd.DataFrame(feature_rows).to_csv(output / "features.csv", index=False)
    results, summaries = {}, {}
    for method in ("ewma", "structural"):
        print(f"Running {method} walk-forward model...", flush=True)
        model = run_model(prices, exposures, specific_features, config, method)
        execution = execution_backtest(prices, model["targets"], exposures,
                                       model["factor_estimates"], model["residual_estimates"], config)
        model.update(execution)
        results[method] = model
        summary = performance_summary(model["daily"], config.trading_days)
        summary["one_step_model_mahalanobis_per_asset"] = float(model["risk_evaluation"]["mahalanobis_per_asset"].mean())
        summary["one_step_model_nll_per_asset"] = float(model["risk_evaluation"]["gaussian_nll_per_asset"].mean())
        summary["max_factor_exposure_error"] = float(model["risk_evaluation"]["max_exposure_error"].max())
        summary["mean_clipped_factor_eigenvalues"] = float(model["risk_evaluation"]["clipped_factor_eigenvalues"].mean())
        summary["leverage_floor_activations"] = int(model["risk_evaluation"]["leverage_floor_activations"].sum())
        if not model["fit_diagnostics"].empty:
            summary["optimizer_fallbacks"] = int(model["fit_diagnostics"]["fallback_to_previous_parameters"].sum())
            summary["fits_with_boundary_hits"] = int((model["fit_diagnostics"]["boundary_hits"] > 0).sum())
        summaries[method] = summary
        save_model_outputs(model, output / method)
    # Long-only context benchmark uses the same execution dates and fee model.
    benchmark_targets = results["ewma"]["targets"].copy()
    benchmark_targets.loc[:, :] = 1.0 / len(config.assets)
    benchmark = execution_backtest(prices, benchmark_targets, exposures,
                                   results["ewma"]["factor_estimates"],
                                   results["ewma"]["residual_estimates"], config)
    results["equal_weight_reference"] = benchmark
    summaries["equal_weight_reference"] = performance_summary(benchmark["daily"], config.trading_days)
    reference_dir = output / "equal_weight_reference"
    reference_dir.mkdir(exist_ok=True)
    benchmark["daily"].to_csv(reference_dir / "daily.csv")
    comparison = pd.DataFrame(summaries).T
    comparison.to_csv(output / "performance_summary.csv", index_label="model")
    make_project_plots(results, results["structural"]["factor_returns"], latest_X, output)
    run_info = {
        "config": asdict(config), "data_provenance": metadata,
        "specific_risk_estimator": f"bounded penalized profile {config.fit_method.upper()}",
        "risk_group_definition": "Two static analyst-defined industry proxies; not historical official classifications.",
        "execution": "Signal after close s; trade at close s+1; earn returns from s+1 to s+2.",
        "risk_forecast_timing": "Risk estimates refer to the signal-date next-close interval; live execution adds one trading-day delay.",
        "cash_rate": 0.0,
        "limitations": ["Five currently surviving stocks; not a historical all-stock universe.",
                        "Three factors leave two residual degrees of freedom per cross-section.",
                        "Regularization, bounds, variance floors, and noise correction are implementation choices.",
                        "Adjusted close execution is an approximation; fees and borrow availability are simplified.",
                        "The momentum direction is fixed in advance; no return-maximizing parameter search is performed.",
                        "Vendor-adjusted history is not a strict point-in-time data archive."],
    }
    (output / "run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    print("\nPerformance summary (fixed teaching universe; not an alpha claim):")
    print(comparison.round(4).to_string())
    print(f"\nResults saved to: {output}")


if __name__ == "__main__":
    main()

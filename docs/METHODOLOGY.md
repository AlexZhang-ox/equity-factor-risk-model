# Methodology

This document describes the implemented research workflow and its assumptions. Standard factor-model identities are distinguished from the choices used to make a small example executable. The repository does not redistribute private source notes.

## 1. Data and characteristics

The fixed universe is AAPL, MSFT, AMZN, GOOGL, and JPM. The default request is `2019-01-01 <= date < 2026-01-01`. Yahoo Finance is accessed through `yfinance` with `auto_adjust=True`. The original local snapshot has 1,760 complete daily rows, from 2019-01-02 through 2025-12-31.

Let `n = 5` and `k = 3`. At the end of date `t`, the exposure matrix `X_t` contains:

1. `common`: a column of ones, interpreted as a common intercept rather than a market-index beta.
2. `momentum`: the cross-sectional z-score of the past 20-close price return.
3. `volatility`: the cross-sectional z-score of the standard deviation of the past 20 daily returns.

The one-column specific-risk characteristic `Z_t` is the cross-sectional z-score of log trailing volatility. All features use observations available through their own date. The code rejects missing prices, zero cross-sectional feature dispersion, and rank-deficient exposure matrices.

Four stocks share the analyst-defined group `TechnologyAndInternet`; JPM is assigned to `Financials`. These are static risk proxies, not reconstructed historical industry memberships. They only enter the specific-risk model. No claim of industry neutrality follows from them.

## 2. Return model and factor portfolios

The return model is

\[
R_{t,t+1} = X_t F_{t,t+1} + \varepsilon_{t,t+1}.
\]

The factor-model assumptions include conditionally mean-zero residuals, diagonal residual covariance `Delta_t`, and zero conditional covariance between factors and residuals. A forecast `Delta_t` is needed before constructing factor portfolios.

Given positive diagonal forecast variances, the implementation computes

\[
W_t = (X_t^\top \Delta_t^{-1} X_t)^{-1} X_t^\top \Delta_t^{-1},
\qquad W_t X_t = I_k.
\]

`W_t` has `k` rows and `n` columns. Each row is a stock portfolio with unit exposure to its target characteristic and zero exposure to the others. These are minimum modeled specific-variance portfolios under the exposure constraints. Unit exposure is not a promise of unit return or unit volatility. Stock weights can be negative and do not generally sum to one.

After the next return is observed,

\[
\widehat F_{t,t+1} = W_t R_{t,t+1}
= F_{t,t+1} + W_t\varepsilon_{t,t+1}.
\]

Thus estimated factor returns contain residual estimation noise. The equality with a realized portfolio return does not make the latent factor directly observable or turn estimation into a forecast of the same period's returns. Residuals are `R - X Fhat`.

## 3. Specific-risk estimation

Both methods process observations chronologically. Previously generated forecasts and weights are retained rather than recomputed with later parameter fits. Training and covariance windows default to 252 completed periods.

### Residual EWMA baseline

The baseline starts from trailing stock variances and updates an individual variance state using historical squared residuals. Because a regression has already fitted the return, residual squares are adjusted by `1 - h_ii`, where `h_ii` is the diagonal of `X W`. The adjustment denominator has a floor of 0.05. The EWMA half-life is 42 periods, and forecast variances are clipped between `1e-6` and `0.01`.

These are stabilization choices. A leverage correction using estimated risk weights is a plug-in adjustment, not a guarantee that small-sample estimation error disappears.

### Structural model

The structural specification is

\[
\Delta_{i,t} = \sigma_t^2 c_{g(i)} \exp(\gamma^\top Z_{i,t}).
\]

The first group multiplier is fixed at one to identify the overall scale. Group log-multipliers and `gamma` are shared across the trailing window, while factor returns and a variance scale are profiled separately for each period. The structural parameters are refitted every 21 periods; the historical scale sequence is combined using EWMA weights to forecast the next scale.

Default fitting uses bounded, penalized profile restricted maximum likelihood, or REML. Gaussian profile MLE is available with `--fit-method mle`. Both modes retain bounds of `[-1, 1]` on fitted log-parameters and a quadratic penalty with coefficient 0.10. Neither mode should be described as unconstrained, unpenalized joint MLE.

There are only `n - k = 2` residual degrees of freedom in each cross-section. REML uses this residual count in the profiled likelihood scale. MLE uses `n` for its likelihood scale, but the risk-forecast scale explicitly uses `n - k` in both modes. The same variance floors and ceilings apply. Parameter sharing, regularization, and clipping are deliberate responses to the small sample, not consequences of the linear return identity.

Optimization status and parameter-bound hits are recorded. A failed optimization retains the preceding parameter vector and records that fallback; it does not switch to simulated prices or silently discard the observation. Boundary hits in the optional MLE run are an estimation diagnostic, not evidence of parameter certainty.

## 4. Two factor covariance matrices

The covariance of investable estimated factor returns differs from the covariance of latent factor returns. At a given date, the model implies

\[
\operatorname{Var}(\widehat F) = \Sigma_F + W\Delta W^\top.
\]

The implementation retains both an empirical factor-portfolio covariance and a noise-corrected latent-factor approximation. With normalized exponential weights `p_s`, let

\[
\bar F = \sum_s p_s \widehat F_s,\qquad
C_{\mathrm{emp}} =
\frac{\sum_s p_s(\widehat F_s-\bar F)(\widehat F_s-\bar F)^\top}
{1-\sum_s p_s^2}.
\]

Using historical plug-in noise estimates `A_s = W_s Delta_s W_s'`, the correction is

\[
C_{\mathrm{noise}} =
\frac{\sum_s p_s(1-p_s) A_s}{1-\sum_s p_s^2},\qquad
C_{\mathrm{latent,raw}} = C_{\mathrm{emp}} - C_{\mathrm{noise}}.
\]

This weighting accounts for estimation of the weighted mean. The correction assumes serially uncorrelated estimation noise, appropriate factor/residual orthogonality, and a locally stable mean. It depends on the accuracy of the estimated specific risk. It is an approximation, not exact recovery of the latent process.

The factor covariance half-life is 63 periods. Positive-semidefinite projection and fixed 15% shrinkage toward the diagonal regularize the empirical and latent estimates. Projection counts and noise ratios are retained as diagnostics.

The empirical covariance describes the estimated factor portfolios and is used for preliminary allocation. The latent approximation is used in the stock risk forecast:

\[
\Sigma_{\mathrm{stock},t}
= X_t\Sigma_{\mathrm{latent},t}X_t^\top + \Delta_t.
\]

This avoids treating all noise in estimated factor returns as additional latent common risk. Full portfolio risk is evaluated as `w' Sigma_stock w`, rather than using the factor component alone.

## 5. Portfolio allocation

Factor-portfolio weights are building blocks; they do not uniquely determine a final stock portfolio. The example supplies the explicit direction vector `(0, 1, 0)` for `(common, momentum, volatility)` before evaluating portfolio returns.

Directions are first normalized by empirical factor-portfolio volatility and scaled. Stock weights are formed as `w = W' a`, then uniformly scaled using full stock risk and the following limits:

| Setting | Default |
| --- | ---: |
| Annualized model volatility target | 10% |
| Gross stock exposure limit | 1.0 |
| Absolute individual stock weight limit | 0.40 |
| Rebalance interval | 5 eligible signal dates |
| Initial capital | 100000 |

Inverse-volatility normalization is not exact equal-risk contribution for correlated factors. With only momentum enabled, this example does not optimize a multi-factor allocation. No expected-return forecast is used to select directions. A trailing estimated factor mean is used to center model risk diagnostics, not to choose the portfolio's sign.

Uniform scaling preserves the signal-date exposure structure. The common intercept constraint makes the default target dollar neutral, up to numerical precision. Exposure and individual-position limits apply to post-trade targets, not continuously to holdings between rebalances. A binding cap can leave predicted volatility below 10%.

## 6. Execution, cash, and costs

A signal computed after close `s` is executed at close `s+1`. It first earns the return from close `s+1` to close `s+2`. This explicit delay prevents using an observed close while assuming an instantaneous trade at that same close.

The backtest keeps stock positions and cash in dollars. For each valuation date, it marks existing positions, charges borrowing on the preceding short notional, and then processes any scheduled rebalance. Before a rebalance, old positions have already drifted with returns.

For pre-trade equity `E_pre`, drifted stock values `v`, target weights `w`, and transaction rate `c`, post-trade equity solves

\[
E_{\mathrm{post}}
= E_{\mathrm{pre}} - c\sum_i|w_i E_{\mathrm{post}} - v_i|.
\]

New dollar holdings are `w * E_post`; cash is the balancing amount. This includes the fee in the target-weight denominator. The first entry is charged, and invalid or insolvent states raise explicit errors.

Transaction cost defaults to 10 bps of full absolute traded notional. Reported turnover is that full notional divided by pre-trade equity; it is not halved. Short borrowing defaults to 3% annualized using 252 trading periods. Cash earns zero. Borrow availability, changing borrow fees, market impact, and executable closing prices are simplified.

Final positions remain marked and open by default. `--liquidate-at-end` instead liquidates at the final close and charges the corresponding fee. Scheduled signals with no next closing price are returned as pending rather than executed at their own signal close.

## 7. Attribution and evaluation

Daily attribution uses actual stock dollar holdings over each return interval, characteristics known at that interval's start, and factor/residual returns estimated after its end. Factor PnL, specific PnL, transaction fees, and borrowing fees sum to the actual daily change in NAV. This is an ex-post explanation, not information available when the earlier signal was formed.

One-step risk scores compare each signal-date stock forecast with its next-close realized return. They do not validate the additional delayed execution interval. The reported Mahalanobis and Gaussian likelihood scores are model diagnostics and should not be presented as proof of calibrated executed-portfolio risk.

All comparison ledgers share the same first execution and valuation dates. The instantaneous entry fee is included without adding a model-warmup sequence of zero returns. For return statistics, that entry fee is combined with the first earned interval; annualization uses 252 periods per year. The cash rate in the Sharpe calculation is zero.

The equal-weight long-only reference uses the same execution and cost conventions, but different gross/net exposure and risk. It provides sample context rather than a risk-matched strategy benchmark.

Plot conventions:

- The factor diagnostic's cumulative series is a sum of returns per unit factor exposure. It is not compounded capital wealth.
- The plotted gross-return index compounds daily gross returns measured against the actual portfolio NAV. It is not a separately rerun zero-cost counterfactual portfolio.
- Final marked holdings may have drifted since the latest rebalance; multiplying them by the latest `X` need not reproduce their original target exposures.

## 8. Validation and limits

Local checks cover factor exposures, return reconstruction, covariance/noise identities, residual leverage correction, stock-risk scaling, delayed execution, and exact dollar accounting. Additional recorded checks compare earlier forecasts with a truncated future dataset. Synthetic data are confined to mathematical tests.

The optional GitHub Actions workflow only runs offline self-tests after installing dependencies. It does not download Yahoo prices or establish historical profitability. The checked-in validation summary records local preparation; it does not claim that GitHub CI has already run.

The five-stock case study has survivorship and selection bias, two residual degrees of freedom, static risk groups, vendor revisions, and approximate execution. Bounds, penalties, floors, and covariance projection affect the estimates. The recorded long-short returns are negative after costs, and no return-maximizing hyperparameter search was performed.

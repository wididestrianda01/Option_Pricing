# Project Spec: Black-Scholes Options Pricing & Greeks Engine (P0)

## Objective  → UPDATED
Implement, from first principles, a complete European-options pricing and risk toolkit — Black-Scholes closed form, full Greeks, implied-volatility inversion, volatility smile, discrete delta-hedging, and a CRR binomial cross-check — applied to the OMXS30 index. Demonstrates command of the standard derivatives toolkit and the independent-verification mindset (own implementation cross-checked against binomial convergence, finite-difference Greeks, and QuantLib).

**Project identity: pre-thesis outreach hook, NOT the thesis.** P0 is a lightweight (~12–14h) deliverable to open SEB Front Office Model Validation / RVS conversations and close the derivatives instrument-class gap. It is intentionally *not* a research contribution. Thesis-grade ambition is routed to **P6 (LGD/EAD IRB)** and **P10 (IFRS-9 / ECL)** — a far stronger fit for the Risk Analyst target. Do not scope-creep P0 into a thesis vehicle.

## Target Role Alignment
| Role | Strength | Signals |
|---|---|---|
| **Quant Analyst** | Strong | Closed-form and lattice pricing, IV inversion, Greeks, from-scratch numerical implementation |
| **Financial Risk Analyst** | Strong | Delta-hedging P&L, sensitivity (Greeks) analysis, discretisation/hedging error, model-validation framing |
| **Finance Data Scientist** | Indirect | Numerical methods (Brent root-finding, finite differences), surface fitting, reproducible pipeline |
| **Portfolio / Asset Mgmt** | Weak | Options as a hedging/overlay instrument; secondary relevance |

> Honest note: P0 is a **credibility/gap-closer** for the quant + market-risk (SEB Front Office Model Validation / RVS) track, not a core hit for the data-science or asset-management aspirations. It is worth the ~12–18h for that specific gap, not more.

## Datasets
- **Underlying (real):** OMXS30 daily close, ticker `^OMXS30` via `yfinance`, or `NASDAQOMXS30` via FRED. Time range: 3–5y for realised-vol estimation and hedging backtest.
- **Risk-free rate:** Swedish T-bill / STIBOR or a flat proxy `r` (documented constant); sensitivity to `r` shown.
- **Option chain (constructed):** Free per-strike OMXS30 option prices / implied vols are **not publicly available** (Nasdaq Nordic options data is behind paid feeds). Chain is therefore **synthetically generated under a specified volatility surface** (base vol + skew + term structure) around the real spot, then priced with BS. This is deliberate: recovering the input surface via the IV solver *is* the validation test.
- **Fallback:** If `yfinance`/FRED is offline, use a synthetic GBM spot path with fixed seed. No behavioural change downstream.

## Methodology

**Black-Scholes (European, no dividends):**
$$C = S_0 N(d_1) - K e^{-rT} N(d_2), \qquad P = K e^{-rT} N(-d_2) - S_0 N(-d_1)$$
$$d_1 = \frac{\ln(S_0/K) + (r + \tfrac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}, \qquad d_2 = d_1 - \sigma\sqrt{T}$$
**Put-call parity check:** $C - P = S_0 - K e^{-rT}$.

**Greeks** (analytical, cross-checked by central finite difference):
$$\Delta_{\text{call}} = N(d_1), \quad \Gamma = \frac{\phi(d_1)}{S_0 \sigma \sqrt{T}}, \quad \mathcal{V} = S_0 \phi(d_1)\sqrt{T}, \quad \Theta, \ \rho$$

**Implied volatility:** solve $C_{\text{BS}}(\sigma) = C_{\text{obs}}$ for $\sigma$ via Brent's method on a bracketed interval, with a Vega-guard fallback (Newton) and handling for near-zero-Vega deep ITM/OTM quotes.

**Volatility smile/surface:** invert IV across strikes and maturities; plot 2D smile slices and a 3D IV surface; comment on skew vs the flat-vol BS assumption.

**CRR binomial tree** (convergence cross-check + American exercise):
$$u = e^{\sigma\sqrt{\Delta t}}, \quad d = 1/u, \quad p = \frac{e^{r\Delta t} - d}{u - d}$$
Show price $\to$ BS as $N \to \infty$.

**Delta-hedging simulation:** discretely rebalanced replicating portfolio along a GBM (or realised OMXS30) path; track hedging-error P&L vs BS theoretical as a function of rebalance frequency (daily / weekly / monthly). Frequency is defined in trading days over a 252-trading-day year: `daily`=1, `weekly`=5, `monthly`=21. Integer `freq` values are interpreted directly as a trading-day step count, bypassing the named presets.

**Testing (pytest, 80% coverage target):**
- `test_pricer.py`: BS price against known reference values; put-call parity to float tolerance; analytic vs central-diff Greeks agreement; IV solver round-trip (price→σ→price) on synthetic quotes.
- `test_binomial.py`: CRR→BS convergence (error shrinks monotonically with N); European CRR matches BS within tolerance at N≥500; cross-check against QuantLib's binomial/analytic BS price.
- `test_smile.py`: `invert_iv_surface` self-recovery error below threshold on the synthetic surface.
- `test_hedge_sim.py`: GBM path generator reproducible under fixed seed; hedge error shrinks monotonically as rebalance frequency increases.
Tests live in `tests/`, run via `pytest --cov=src --cov-report=term-missing`.

**Key assumptions & limitations:** constant `r` and `σ` (BS), no dividends (OMXS30 is a price index — flag this), no transaction costs in the base hedge, geometric-Brownian dynamics, European exercise for closed form. Synthetic chain means smile shape is *assumed*, not market-observed — stated explicitly.

## Deliverables
- [ ] Jupyter Notebook (primary) — `black-scholes-pricing-greeks.ipynb`
- [ ] README.md — project, data sources, OMXS30 chain caveat, run instructions
- [ ] requirements.txt — pinned (`numpy`, `scipy`, `pandas`, `matplotlib`, `yfinance`, `QuantLib`)
- [ ] src/ main engine — `pricer.py` (BS + Greeks + IV), `binomial.py` (CRR), `smile.py` (surface), `hedge_sim.py` (delta hedge)

## Notebook Structure
1. Introduction & motivation — options, BS assumptions, why validation framing
2. Data acquisition & cleaning — OMXS30 underlying, rate proxy, synthetic-chain construction
3. EDA with visualizations — spot path, realised vol, return distribution vs lognormal
4. Methodology / model implementation — BS pricer, Greeks (analytic vs finite-diff), IV solver, binomial, hedge sim
5. Results & interpretation — parity check, Greeks behaviour, smile/surface, binomial convergence, hedging-error vs frequency
6. Limitations & extensions — dividends, stochastic vol (Heston), American via LSM, real chain if licensed

## Success Criteria
- `Restart & Run All` clean, end-to-end, seeded.
- Put-call parity holds to floating-point tolerance; analytic Greeks match finite-difference within tolerance.
- IV solver recovers the input synthetic surface (max abs error reported).
- Binomial price converges to BS (convergence plot).
- Delta-hedging error shrinks monotonically with rebalance frequency (quantified, not just asserted).
- Every section opens with what/why markdown; every output interpreted in plain English.
- At least one non-trivial result with interpretation (hedging-error-vs-frequency table).

## Estimated Complexity  → UPDATED
**Medium.** Six components (BS+Greeks, IV solver, smile/surface, binomial, delta-hedge sim, validation cross-checks). Target effort **~12–14h** consistent with the outreach-hook identity. **Hard scope cap — no thesis-track extensions in P0** (no Heston/local-vol benchmark, no misspecified-process hedging distribution, no SSVI arbitrage-free fitting; those belong in a thesis, not here). If time overruns, cut the delta-hedge simulation first; the IV-solver + smile + Greeks cross-check block is the highest-signal core to protect.

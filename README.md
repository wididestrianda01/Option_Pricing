# Black-Scholes Options Pricing & Greeks Engine

European-options pricing and risk toolkit built from first principles and validated against independent methods — Black-Scholes closed form, full Greeks (analytical + finite-difference), implied-volatility inversion, synthetic volatility smile/surface, CRR binomial cross-check, American early-exercise premium, and discrete delta-hedging P&L simulation — calibrated to the Swedish **OMXS30** index.

**Primary deliverable:** [`black-scholes-pricing-greeks.ipynb`](black-scholes-pricing-greeks.ipynb)

## Motivation

Production pricing libraries are only trusted after systematic validation. This project adopts a **Validation-Driven Design**: every pricing and risk component is cross-checked against an independent method, forming a validation ladder from model-free to cross-model checks:

1. **Model-free** — put-call parity across strikes (must hold at machine precision)
2. **Intra-model** — analytical Greeks vs. central finite differences
3. **Inversion** — implied-volatility solver round-trips prices back to input vols
4. **Cross-model** — CRR binomial tree converges to the Black-Scholes closed form
5. **Dynamic replication** — discrete delta-hedging error obeys the $1/\sqrt{N}$ scaling law

## Key Results

| Check | Result | Tolerance | Status |
| :--- | :--- | :--- | :---: |
| Put-call parity max error | $5.7 \times 10^{-13}$ | $< 10^{-8}$ | ✅ |
| Analytical vs. FD Greeks max rel. error | $< 0.001\%$ | $< 0.1\%$ | ✅ |
| IV self-recovery max error (full strike/maturity grid) | $9.3 \times 10^{-11}$ | $< 10^{-6}$ | ✅ |
| CRR convergence error at $N = 500$ | $0.05$ | $< 0.1$ | ✅ |
| Hedge error monotonicity (mean abs, SEK) | daily $10.3$ < weekly $23.8$ < monthly $50.4$ | monotone | ✅ |

Selected findings:

- **$\sqrt{N}$ hedging rule confirmed:** moving from monthly (12 rebalances) to daily (252) reduces mean hedge error by a factor of $\approx 4.9\times$, vs. the theoretical $\sqrt{21} \approx 4.6\times$. Hedge error concentrates ATM, where gamma peaks.
- **American early-exercise premium:** an ATM 1-year American put on OMXS30 carries a premium of $\approx 4.4\%$ over its European counterpart; the American call premium is exactly zero (no dividends), as theory requires.
- **Binomial convergence is $O(1/N)$ but oscillatory** (strike/node parity effect), motivating parity smoothing in production lattices.
- **EDA on OMXS30** (5y daily): volatility clustering (rolling vol 10–30% around $\hat\sigma \approx 16.8\%$), negative skew ($\approx -0.16$), and fat tails — direct empirical violations of constant-vol GBM that motivate the volatility smile modeled in the notebook.

## Data Sources

- **Underlying:** OMXS30 daily closes via `yfinance` (`^OMX`), ~5y history. Automatic fallback to a fixed-seed synthetic GBM path so `Restart & Run All` never fails offline.
- **Risk-free rate:** flat proxy $r$ (documented constant), with STIBOR/T-bill sensitivity discussed.
- **Option chain: synthetic.** Strike-level OMXS30 option data is not freely available (Nasdaq Nordic feeds are paid). A chain is generated from a known volatility surface $\sigma(K,T) = \sigma_{\text{base}} + \text{skew} \cdot (K/S_0 - 1) + \text{term} \cdot \sqrt{T}$. This gives an exact ground truth: recovering the surface via the IV solver becomes a precise, noise-free test of solver correctness. The smile shape is therefore **assumed, not market-observed**.

## Repository Structure

```
├── black-scholes-pricing-greeks.ipynb   # primary deliverable
├── src/
│   ├── pricer.py        # BS closed form, analytical + FD Greeks, Brent–Newton IV solver (with vega guard)
│   ├── binomial.py      # vectorized CRR tree, convergence study, American early-exercise premium
│   ├── smile.py         # synthetic vol surface, IV self-recovery, skew/term-structure analysis
│   └── hedge_sim.py     # GBM path generation, discrete delta-rebalancing, hedge-error P&L analysis
├── tests/               # pytest suite (parity, Greeks, IV, CRR vs. QuantLib, hedging convergence)
├── requirements.txt
└── README.md
```

## How to Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# run test suite with coverage
pytest --cov=src --cov-report=term-missing

# run the notebook end-to-end
jupyter notebook black-scholes-pricing-greeks.ipynb   # Kernel → Restart & Run All
```

Requires Python 3.10+. Key dependencies: `numpy`, `scipy`, `pandas`, `matplotlib`, `yfinance`, `QuantLib` (used only as an independent cross-check in tests), `pytest`.

Reproducibility: all stochastic components use fixed seeds (`seed=42` default); the notebook runs deterministically offline via the GBM fallback.

## Methodology Highlights

- **IV solver:** hybrid Brent (guaranteed-convergence bracketing) + Newton-Raphson refinement with a **vega guard** to prevent divergence in deep ITM/OTM wings where vega $\to 0$; arbitrage-violating prices return `None`.
- **Finite differences:** central scheme with $h = 10^{-4} S_0$, chosen to balance $O(h^2)$ truncation error against floating-point cancellation.
- **Hedging simulation:** self-financing replicating portfolio (delta position + cash account accruing at $r$), rebalanced daily/weekly/monthly across a strike grid; hedge error = terminal portfolio value − option payoff.

## Assumptions & Limitations

Constant $r$ and $\sigma$; no dividends (OMXS30 is a price index); no transaction costs; GBM dynamics (no jumps); European exercise for closed-form pricing (American exercise handled separately via the binomial tree); single-path hedge results are path-dependent — the $1/\sqrt{N}$ law holds in expectation across paths (verified over 40 seeds in tests).

Industry extensions discussed in the notebook: local/stochastic volatility (Dupire, Heston) and GARCH for the smile and clustering; Hull-White / bootstrapped STIBOR curves for stochastic rates; Leland correction and deep hedging for transaction costs; jump-diffusion for tail risk.

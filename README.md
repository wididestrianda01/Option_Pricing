# Black-Scholes Options Pricing & Greeks Engine

European-options pricing and risk toolkit built from first principles — Black-Scholes
closed form, full Greeks (analytical + finite-difference), implied-volatility inversion,
synthetic volatility smile/surface, CRR binomial cross-check, and discrete delta-hedging
P&L simulation — applied to the OMXS30 index.

## Data sources
- **Underlying:** OMXS30 daily close via `yfinance` (`^OMXS30`), 3–5y history. Falls back
  to a fixed-seed synthetic GBM path if offline.
- **Risk-free rate:** flat proxy `r` (documented constant); Swedish STIBOR/T-bill sensitivity shown.
- **Option chain:** **synthetic** — free per-strike OMXS30 option data is not publicly
  available (Nasdaq Nordic feeds are paid). A chain is generated under an assumed
  volatility surface (base + skew + term structure); recovering that surface via the
  IV solver is itself the validation test. This is **not** a market-observed smile.

## Run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest --cov=src --cov-report=term-missing
jupyter notebook black-scholes-pricing-greeks.ipynb  # Restart & Run All
```

## Structure
- `src/pricer.py` — BS price, Greeks, IV solver
- `src/binomial.py` — CRR tree, convergence, American premium
- `src/smile.py` — synthetic surface, IV self-recovery, skew analysis
- `src/hedge_sim.py` — GBM path, delta-rebalance hedge simulation
- `tests/` — pytest suite, 80% coverage target

## Key assumptions & limitations
Constant `r` and `σ` (flat Black-Scholes); no dividends (OMXS30 is a price index); no
transaction costs in the base hedge; geometric Brownian motion; European exercise for
closed-form pricing (American analysed separately via the binomial tree). The synthetic
chain means the smile shape is assumed, not market-observed.

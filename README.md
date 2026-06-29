# JPMorgan Chase & Co. — Quantitative Research Job Simulation

This repository contains my completed work for the [JPMorgan Chase Quantitative Research Job Simulation](https://www.theforage.com/simulations/jpmorgan/quantitative-research-qfjp) hosted on Forage. The simulation covers four tasks that mirror real quantitative research work done at JPMorgan, spanning commodity pricing, derivatives valuation, credit risk modelling, and data quantization.

---

## Tasks

### Task 1 — Natural Gas Price Estimation (`nat_gas_pricer.py`)
Analysed monthly natural gas price data from October 2020 to September 2024 to build a price estimation model. Uses cubic spline interpolation within the data range and a trend-plus-seasonal decomposition model to extrapolate prices up to one year beyond the data. Exposes a simple `get_price(date)` API and produces visualisations of the seasonal patterns and year-over-year price behaviour.

### Task 2 — Gas Storage Contract Pricing (`gas_storage_pricer.py`)
Built a prototype pricing model for natural gas storage contracts. The function `price_storage_contract()` takes injection and withdrawal dates, volumes, and prices alongside facility parameters — storage cost, injection/withdrawal rates, and transport costs — and returns the net contract value with a full cash flow breakdown. Validated against example contracts including multi-date schedules and constraint violations.

### Task 3 — Loan Default Prediction (`loan_default_model.py`)
Trained a Random Forest classifier on a portfolio of 10,000 borrowers to predict the probability of default (PD). Engineered features including debt-to-income ratio, loan-to-income ratio, and credit utilisation. The model achieved a ROC-AUC of 0.9997 on the test set. The `expected_loss()` function takes borrower details and returns the expected loss in dollars (PD × LGD × loan amount), alongside a risk flag — LOW, MEDIUM, or HIGH — based on configurable PD thresholds.

### Task 4 — FICO Score Bucketing (`fico_bucketing.py`)
Implemented a dynamic programming solution to optimally partition FICO scores into rating buckets by maximising a log-likelihood function that accounts for default density within each bucket. The approach is fully general — `get_rating(fico_score, n_buckets)` accepts any number of buckets and any dataset, making it reusable for future mortgage portfolios.

---

## Setup

```bash
git clone https://github.com/kaimanareal-cloud/jpmorgan-quant-research-simulation.git
cd jpmorgan-quant-research-simulation
pip install -r requirements.txt
```

## Requirements
- Python 3.9+
- pandas, numpy, scipy, scikit-learn, statsmodels, matplotlib, python-dateutil

---

*Completed as part of the Forage JPMorgan Chase Quantitative Research Virtual Experience Programme.*

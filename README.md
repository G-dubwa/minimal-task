# Deep Hedging: Neural Network Hedge Portfolios

**MFE Research Project 1 — UCT 2026**

Neural network that simultaneously learns an option premium and a dynamic hedging strategy for a European call in the Black-Scholes model, following [Buehler et al. (2019) *Deep Hedging*](https://www.tandfonline.com/doi/full/10.1080/14697688.2019.1571683).

## Results (minimal task)

| | |
|---|---|
| Black-Scholes price C₀ | 0.16411 |
| Learned premium π | 0.16413 |
| Hedged P&L std (test) | 0.00804 |
| Unhedged P&L std (test) | 0.23117 |

The network recovers the Black-Scholes price to 4 decimal places and reduces P&L volatility by **28×** vs. the unhedged position.

## Model parameters

| Parameter | Value |
|---|---|
| Risk-free rate r | 0.0 |
| Volatility σ | 0.4 |
| Initial price S₀ | 1.0 |
| Strike K | 0.9 |
| Maturity T | 0.5 yr |
| Hedging steps N | 125 (≈ daily) |
| Training paths M | 100 000 |

## Architecture

- **Premium subnetwork** — input `0` → 2× Dense(32, tanh) → Dense(1, linear) → π
- **Delta subnetwork** (weight-sharing) — input `[Sₙ, τₙ]` → 2× Dense(32, tanh) → Dense(1, sigmoid) → δₙ ∈ [0, 1]
- **Portfolio** — V₀ = π, Vₙ₊₁ = Vₙ + δₙ(Sₙ₊₁ − Sₙ)
- **Loss** — MSE[Φ(Sₙ) − Vₙ] minimised with Adam

All N=125 delta calls are batched into a single `(M×N, 2)` forward pass rather than unrolled, keeping the graph small and training fast (~9 s/epoch on Apple M1).

## Setup

```bash
conda create -n deep_hedging python=3.11 -y
conda activate deep_hedging
pip install tensorflow==2.17.1 tensorflow-metal==1.2.0 numpy scipy matplotlib ipykernel
python -m ipykernel install --user --name deep_hedging --display-name "Deep Hedging (TF 2.17)"
```

> **Google Colab**: upload `deep_hedging.ipynb`, set runtime to GPU, and run all cells — no setup needed.

## Usage

```bash
conda activate deep_hedging
python deep_hedging.py
```

Outputs saved to the project directory:

| File | Description |
|---|---|
| `learning_curve.png` | Train/validation MSE per epoch |
| `pnl_histograms.png` | Hedged vs. unhedged P&L distributions |
| `hedge_path_in_the_money.png` | NN vs. BS delta along an ITM path |
| `hedge_path_out_of_the_money.png` | NN vs. BS delta along an OTM path |
| `best_model.keras` | Saved model weights (best validation loss) |

## Files

```
.
├── deep_hedging.py       # standalone script
├── deep_hedging.ipynb    # Jupyter / Colab notebook
└── README.md
```

## Reference

Buehler, H., Gonon, L., Teichmann, J., & Wood, B. (2019). Deep hedging. *Quantitative Finance*, 19(8), 1271–1291.
# minimal-task

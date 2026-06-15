"""
Deep Hedging: Neural Network Hedge Portfolio for a Vanilla European Call
========================================================================
Minimal task — MFE Research Project 1, UCT 2026

Model:   Black-Scholes  (r=0, σ=0.4, S0=1, K=0.9, T=0.5)
Network: Weight-sharing delta subnetwork  δ_n = NN(S_n, τ_n)
         + premium subnetwork  π = NN(0)
Loss:    MSE[ V_N − Φ(S_N) ]   where  V_N = π + Σ_n δ_n ΔS_n

Reference: Buehler et al. (2019) "Deep Hedging", Quantitative Finance 19(8).
"""

import numpy as np
import tensorflow as tf
import keras
from keras import layers, Model, Input, callbacks, optimizers, ops as kops
import matplotlib.pyplot as plt
from scipy.stats import norm

tf.random.set_seed(0)
np.random.seed(0)

# ══════════════════════════════════════════════════════════════════
# 1.  PARAMETERS
# ══════════════════════════════════════════════════════════════════
r      = 0.0    # continuously-compounded risk-free rate
sigma  = 0.4    # Black-Scholes volatility
S0     = 1.0    # initial stock price
K      = 0.9    # strike
T      = 0.5    # maturity (years)
N      = 125    # trading steps  (≈ daily for a 6-month option)
dt     = T / N

M_TRAIN = 100_000   # training paths
M_TEST  =  10_000   # test paths

# Network hyper-parameters
N_UNITS  = 32       # neurons per hidden layer
N_LAYERS = 2        # hidden layers in each subnetwork
ACT      = 'tanh'   # activation (tanh works well for financial features)

BATCH_SIZE    = 256
EPOCHS        = 50
LEARNING_RATE = 1e-3

# ══════════════════════════════════════════════════════════════════
# 2.  BLACK-SCHOLES REFERENCE FORMULAS
# ══════════════════════════════════════════════════════════════════

def bs_call_price(S, K, T, r, sigma):
    """Analytic Black-Scholes European call price."""
    if T <= 1e-10:
        return float(max(S - K, 0.0))
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))

def bs_call_delta(S, K, tau, r, sigma):
    """Vectorised Black-Scholes call delta.  tau = time to maturity."""
    tau = np.maximum(np.asarray(tau, dtype=float), 1e-10)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    return norm.cdf(d1)

C0_bs = bs_call_price(S0, K, T, r, sigma)
print(f"Black-Scholes call price  C0 = {C0_bs:.5f}")

# ══════════════════════════════════════════════════════════════════
# 3.  MONTE CARLO SIMULATION  (GBM paths)
# ══════════════════════════════════════════════════════════════════

def simulate_gbm(M, N, S0, r, sigma, T, seed=None):
    """
    Simulate M paths of a Geometric Brownian Motion with N time steps.
    Returns array of shape (M, N+1).
    """
    rng = np.random.default_rng(seed)
    dt_ = T / N
    Z   = rng.standard_normal((M, N)).astype(np.float32)
    log_inc = np.float32((r - 0.5 * sigma**2) * dt_) + np.float32(sigma * np.sqrt(dt_)) * Z
    log_S   = np.float32(np.log(S0)) + np.cumsum(log_inc, axis=1)
    return np.hstack([np.full((M, 1), S0, dtype=np.float32), np.exp(log_S)])

S_train = simulate_gbm(M_TRAIN, N, S0, r, sigma, T, seed=42)
S_test  = simulate_gbm(M_TEST,  N, S0, r, sigma, T, seed=123)
print(f"Train paths: {S_train.shape},  Test paths: {S_test.shape}")

# ══════════════════════════════════════════════════════════════════
# 4.  NEURAL NETWORK ARCHITECTURE
# ══════════════════════════════════════════════════════════════════

def dense_block(x, n_units, n_layers, act, prefix):
    """Stack of fully-connected layers with the same activation."""
    for k in range(n_layers):
        x = layers.Dense(n_units, activation=act, name=f'{prefix}_h{k}')(x)
    return x


def build_deep_hedging_model(N, K, T, dt, n_units=32, n_layers=2, act='tanh'):
    """
    Build the deep hedging network.

    Architecture
    ─────────────
    Premium  π  :  0  →  Dense block  →  Dense(1, linear)
    Delta  δ_n  :  [S_n, τ_n]  →  Dense block  →  Dense(1, sigmoid)
                   (same weights reused at every time step — weight sharing)

    Portfolio update (r=0, no discounting needed):
        V_0 = π
        V_{n+1} = V_n + δ_n (S_{n+1} − S_n)

    Output: V_N − Φ(S_N)   →   minimise MSE to drive this to 0.

    Parameters
    ──────────
    N      : number of hedging steps
    K      : strike price
    T, dt  : maturity and step size  (used to compute τ_n = T − n·dt)
    """
    # ── Inputs ──────────────────────────────────────────────────────
    S_path = Input(shape=(N + 1,), dtype='float32', name='S_path')

    # Time-to-maturity at each step (fixed, not trainable)
    tau_vals = np.float32(T - np.arange(N) * dt)   # shape (N,)

    # ── Premium subnetwork: learns a constant premium from zeros ───
    zeros = kops.zeros_like(S_path[:, :1])           # (M, 1) of 0s
    h_pi  = dense_block(zeros, n_units, n_layers, act, 'pi')
    pi    = layers.Dense(1, activation='linear', name='pi')(h_pi)   # (M, 1)

    # ── Delta subnetwork (weight-sharing across time) ──────────────
    delta_in  = Input(shape=(2,), dtype='float32', name='delta_in')
    h_dlt     = dense_block(delta_in, n_units, n_layers, act, 'dlt')
    delta_out = layers.Dense(1, activation='sigmoid', name='delta_out')(h_dlt)
    delta_net = Model(inputs=delta_in, outputs=delta_out, name='delta_net')

    # ── Apply delta_net to all N steps in one batched call ─────────
    # Rather than looping N=125 times (slow unrolled graph), we reshape
    # (M, N, 2) → (M*N, 2), call delta_net once, reshape back to (M, N).
    S_steps = S_path[:, :N]                                  # (M, N)
    tau_2d  = kops.ones_like(S_steps) * tau_vals             # (M, N) broadcast
    feat_3d = kops.stack([S_steps, tau_2d], axis=-1)         # (M, N, 2)
    feat_2d = kops.reshape(feat_3d, (-1, 2))                 # (M*N, 2)
    delta_flat = delta_net(feat_2d)                          # (M*N, 1)
    all_deltas = kops.reshape(delta_flat, (-1, N))           # (M, N)

    # ── Self-financing portfolio P&L ───────────────────────────────
    dS  = S_path[:, 1:] - S_path[:, :N]                     # (M, N)
    pnl = kops.sum(all_deltas * dS, axis=1, keepdims=True)  # (M, 1)
    V_N = pi + pnl

    # ── Payoff and hedge error ─────────────────────────────────────
    payoff    = kops.maximum(S_path[:, N:N+1] - K, 0.0)
    hedge_err = V_N - payoff    # target: 0

    train_model = Model(inputs=S_path, outputs=hedge_err, name='deep_hedging')
    return train_model, delta_net


model, delta_net = build_deep_hedging_model(
    N, K, T, dt, n_units=N_UNITS, n_layers=N_LAYERS, act=ACT
)
print(f"Trainable parameters: {model.count_params():,}")

# ══════════════════════════════════════════════════════════════════
# 5.  TRAINING
# ══════════════════════════════════════════════════════════════════

def mse_hedge_loss(y_true, y_pred):
    """MSE of hedge error V_N − Φ(S_N).  y_true is ignored (always zero)."""
    return kops.mean(kops.square(y_pred))


model.compile(
    optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=mse_hedge_loss,
)

zeros_train = np.zeros((M_TRAIN, 1), dtype=np.float32)
zeros_test  = np.zeros((M_TEST,  1), dtype=np.float32)

cb_list = [
    callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5,
        min_lr=1e-5, verbose=1,
    ),
    callbacks.EarlyStopping(
        monitor='val_loss', patience=10,
        restore_best_weights=True,
    ),
    callbacks.ModelCheckpoint(
        'best_model.keras', monitor='val_loss', save_best_only=True,
    ),
]

history = model.fit(
    S_train, zeros_train,
    validation_data=(S_test, zeros_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=cb_list,
    verbose=1,
)

# ══════════════════════════════════════════════════════════════════
# 6.  EXTRACT LEARNED PREMIUM
# ══════════════════════════════════════════════════════════════════
# The premium π does not depend on the stock path at all (its subnetwork
# receives a constant 0 as input).  A path of all-zeros gives dS=0 at
# every step and a zero payoff, so the model output equals π directly.

dummy = np.zeros((1, N + 1), dtype=np.float32)
pi_nn = float(model.predict(dummy, verbose=0)[0, 0])

print(f"\nLearned premium     π  = {pi_nn:.5f}")
print(f"Black-Scholes price C0 = {C0_bs:.5f}")

# ══════════════════════════════════════════════════════════════════
# 7.  P&L ANALYSIS ON THE TEST SET
# ══════════════════════════════════════════════════════════════════

hedge_errors = model.predict(S_test, verbose=0)[:, 0]   # V_N − Φ(S_N)
payoff_test  = np.maximum(S_test[:, N] - K, 0.0)
unhedged_pnl = pi_nn - payoff_test   # seller's P&L without hedging

print("\n── Hedged portfolio (seller) ─────────────────────")
print(f"  Mean   : {np.mean(hedge_errors):+.5f}")
print(f"  Std    : {np.std(hedge_errors):.5f}")
print(f"  1st %  : {np.percentile(hedge_errors,  1):.5f}")
print(f"  5th %  : {np.percentile(hedge_errors,  5):.5f}")
print(f"  RMSE   : {np.sqrt(np.mean(hedge_errors**2)):.5f}")

print("\n── Unhedged (naked short call) ───────────────────")
print(f"  Mean   : {np.mean(unhedged_pnl):+.5f}")
print(f"  Std    : {np.std(unhedged_pnl):.5f}")
print(f"  1st %  : {np.percentile(unhedged_pnl,  1):.5f}")
print(f"  5th %  : {np.percentile(unhedged_pnl,  5):.5f}")

# ══════════════════════════════════════════════════════════════════
# 8.  PLOTS
# ══════════════════════════════════════════════════════════════════

plt.style.use('seaborn-v0_8-whitegrid')

# ── (a) Learning curve ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(history.history['loss'],     label='Train')
ax.plot(history.history['val_loss'], label='Validation')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss (MSE)')
ax.set_title('Learning Curve')
ax.legend()
plt.tight_layout()
plt.savefig('learning_curve.png', dpi=150)
plt.close()
print("Saved: learning_curve.png")

# ── (b) P&L histograms ───────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.hist(hedge_errors,  bins=60, color='steelblue', edgecolor='none')
ax1.set_xlabel(r'$V_N - \Phi(S_N)$')
ax1.set_ylabel('Count')
ax1.set_title('Hedged P&L')

ax2.hist(unhedged_pnl, bins=60, color='steelblue', edgecolor='none')
ax2.set_xlabel(r'$\pi - \Phi(S_N)$')
ax2.set_ylabel('Count')
ax2.set_title('Unhedged P&L (naked short call)')

plt.tight_layout()
plt.savefig('pnl_histograms.png', dpi=150)
plt.close()
print("Saved: pnl_histograms.png")

# ── (c) Step-by-step hedge paths ─────────────────────────────────

def nn_portfolio_path(S_path_1d):
    """Compute NN delta and portfolio value for a single price path."""
    tau_arr = T - np.arange(N) * dt                            # (N,)
    feat    = np.column_stack([S_path_1d[:N],
                               tau_arr]).astype(np.float32)    # (N, 2)
    deltas  = delta_net.predict(feat, verbose=0)[:, 0]         # (N,)
    dS      = np.diff(S_path_1d)                               # (N,)
    V       = np.empty(N + 1)
    V[0]    = pi_nn
    for n in range(N):
        V[n + 1] = V[n] + deltas[n] * dS[n]
    return deltas, V

def bs_portfolio_path(S_path_1d):
    """Black-Scholes delta and portfolio value for a single price path."""
    tau_arr = T - np.arange(N) * dt
    deltas  = bs_call_delta(S_path_1d[:N], K, tau_arr, r, sigma)  # (N,)
    dS      = np.diff(S_path_1d)                                   # (N,)
    V       = np.empty(N + 1)
    V[0]    = C0_bs
    for n in range(N):
        V[n + 1] = V[n] + deltas[n] * dS[n]
    return deltas, V

S_T_test = S_test[:, N]
itm_idx  = np.where(S_T_test > K)[0][0]
otm_idx  = np.where(S_T_test < K)[0][0]
times    = np.linspace(0, T, N + 1)
t_mid    = times[:-1]     # delta is held from t_n to t_{n+1}

for idx, tag in [(itm_idx, 'in_the_money'), (otm_idx, 'out_of_the_money')]:
    S_i = S_test[idx]
    nn_dlt, nn_V  = nn_portfolio_path(S_i)
    bs_dlt, bs_V  = bs_portfolio_path(S_i)
    payoff_i = max(S_i[N] - K, 0.0)

    fig, (ax_v, ax_d) = plt.subplots(1, 2, figsize=(13, 4))
    title_str = tag.replace('_', '-').capitalize()
    fig.suptitle(f'{title_str} path  (S_T = {S_i[N]:.4f},  payoff = {payoff_i:.4f})')

    ax_v.plot(times, S_i,  'k',   lw=1.2, label='Stock price $S_t$')
    ax_v.plot(times, nn_V, 'b',   lw=1.2, label='NN hedge $V_t$')
    ax_v.plot(times, bs_V, 'r--', lw=1.2, label='BS hedge $V_t$')
    ax_v.axhline(payoff_i, color='gray', ls=':', lw=1, label='Payoff')
    ax_v.set_xlabel('Time $t$')
    ax_v.set_ylabel('Value')
    ax_v.set_title('Portfolio value')
    ax_v.legend(fontsize=8)

    ax_d.plot(t_mid, nn_dlt, 'b',   lw=1.2, label='NN $\\delta_n$')
    ax_d.plot(t_mid, bs_dlt, 'r--', lw=1.2, label='BS $\\delta_n$')
    ax_d.set_xlabel('Time $t$')
    ax_d.set_ylabel('Delta')
    ax_d.set_title('Hedging delta')
    ax_d.set_ylim(-0.05, 1.05)
    ax_d.legend(fontsize=8)

    plt.tight_layout()
    fname = f'hedge_path_{tag}.png'
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved: {fname}")

print("\nAll done.")

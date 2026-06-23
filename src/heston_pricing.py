"""Lightweight Heston pricing helpers for submission validation.

The main Heston training notebook contains the production Torch implementation
used for the hedging experiments. This module provides a small NumPy-only
validation implementation so the submission-facing notebook can quickly compare
COS prices with an independent Carr--Madan quadrature without launching the
training workflow.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HestonParams:
    kappa: float = 2.0
    theta: float = 0.16
    xi: float = 0.60
    rho: float = -0.70
    r: float = 0.0


def heston_cf_log_s(u, s0: float, v0: float, tau: float, params: HestonParams):
    """Heston characteristic function for log S_T using the little-trap form."""
    u = np.asarray(u, dtype=np.complex128)
    iu = 1j * u
    kappa, theta, xi, rho, r = (
        params.kappa,
        params.theta,
        params.xi,
        params.rho,
        params.r,
    )
    d = np.sqrt((rho * xi * iu - kappa) ** 2 + xi**2 * (iu + u**2))
    num = kappa - rho * xi * iu - d
    den = kappa - rho * xi * iu + d
    g = num / den
    exp_dt = np.exp(-d * tau)
    d_term = (num / xi**2) * (1.0 - exp_dt) / (1.0 - g * exp_dt)
    c_term = r * iu * tau + (kappa * theta / xi**2) * (
        num * tau - 2.0 * np.log((1.0 - g * exp_dt) / (1.0 - g))
    )
    return np.exp(c_term + d_term * v0 + iu * np.log(s0))


def heston_cos_call(
    s0: float,
    v0: float,
    strike: float,
    tau: float,
    params: HestonParams = HestonParams(),
    n_cos: int = 256,
    a: float = -7.0,
    b: float = 7.0,
) -> float:
    """COS European call price with y = log(S_T / K) on a fixed interval."""
    if tau <= 1e-12:
        return float(max(s0 - strike, 0.0))

    k = np.arange(n_cos, dtype=np.float64)
    u = k * np.pi / (b - a)
    phi_y = heston_cf_log_s(u, s0, v0, tau, params) * np.exp(-1j * u * np.log(strike))

    c, d = 0.0, b
    omega = u
    exp_d = np.exp(d)
    exp_c = np.exp(c)
    chi = (
        np.cos(omega * (d - a)) * exp_d
        - np.cos(omega * (c - a)) * exp_c
        + omega * np.sin(omega * (d - a)) * exp_d
        - omega * np.sin(omega * (c - a)) * exp_c
    ) / (1.0 + omega**2)

    psi = np.empty_like(k, dtype=np.float64)
    psi[0] = d - c
    psi[1:] = (
        np.sin(omega[1:] * (d - a)) - np.sin(omega[1:] * (c - a))
    ) / omega[1:]

    payoff_coeff = 2.0 / (b - a) * (chi - psi)
    payoff_coeff[0] *= 0.5
    price = strike * np.exp(-params.r * tau) * np.real(
        np.sum(phi_y * np.exp(-1j * u * a) * payoff_coeff)
    )
    return float(max(price, 0.0))


def heston_carr_madan_call(
    s0: float,
    v0: float,
    strike: float,
    tau: float,
    params: HestonParams = HestonParams(),
    alpha: float = 1.5,
    u_max: float = 200.0,
    n_grid: int = 20_000,
) -> float:
    """Carr--Madan damped Fourier call price using trapezoidal quadrature."""
    if tau <= 1e-12:
        return float(max(s0 - strike, 0.0))

    u = np.linspace(1e-10, u_max, int(n_grid), dtype=np.float64)
    log_k = np.log(strike)
    z = u - 1j * (alpha + 1.0)
    phi = heston_cf_log_s(z, s0, v0, tau, params)
    denom = alpha**2 + alpha - u**2 + 1j * (2.0 * alpha + 1.0) * u
    psi = np.exp(-params.r * tau) * phi / denom
    integrand = np.real(np.exp(-1j * u * log_k) * psi)
    price = np.exp(-alpha * log_k) / np.pi * np.trapz(integrand, u)
    return float(max(price, 0.0))


def validation_grid(params: HestonParams = HestonParams()) -> tuple[np.ndarray, dict]:
    """Run the report validation grid and return row-level and summary results."""
    rows = []
    for s0 in (0.8, 1.0, 1.2):
        for v0 in (0.08, 0.16, 0.32):
            for tau in (0.5, 0.75, 1.0):
                cos_price = heston_cos_call(s0, v0, 1.0, tau, params)
                cm_price = heston_carr_madan_call(s0, v0, 1.0, tau, params)
                rows.append(
                    {
                        "S": s0,
                        "v": v0,
                        "K": 1.0,
                        "tau": tau,
                        "COS": cos_price,
                        "CarrMadan": cm_price,
                        "abs_error": abs(cos_price - cm_price),
                    }
                )
    table = np.array(rows, dtype=object)
    errors = np.array([row["abs_error"] for row in rows], dtype=np.float64)
    summary = {
        "max_abs_error": float(np.max(errors)),
        "median_abs_error": float(np.median(errors)),
        "n_grid_points": int(len(rows)),
    }
    return table, summary


def finite_difference_greeks(
    s0: float = 1.0,
    v0: float = 0.16,
    strike: float = 1.0,
    tau: float = 1.0,
    h_s: float = 1e-4,
    h_v: float = 1e-5,
    params: HestonParams = HestonParams(),
) -> dict:
    """Central finite-difference delta and dC/dv stability check."""

    def price(s, v):
        return heston_cos_call(s, v, strike, tau, params)

    delta_h = (price(s0 + h_s, v0) - price(s0 - h_s, v0)) / (2.0 * h_s)
    delta_half = (price(s0 + h_s / 2.0, v0) - price(s0 - h_s / 2.0, v0)) / h_s
    dvd_h = (price(s0, v0 + h_v) - price(s0, v0 - h_v)) / (2.0 * h_v)
    dvd_half = (price(s0, v0 + h_v / 2.0) - price(s0, v0 - h_v / 2.0)) / h_v
    return {
        "S": s0,
        "v": v0,
        "K": strike,
        "tau": tau,
        "h_S": h_s,
        "h_v": h_v,
        "delta_fd_h": float(delta_h),
        "delta_fd_h_half": float(delta_half),
        "delta_abs_diff": float(abs(delta_h - delta_half)),
        "dC_dv_fd_h": float(dvd_h),
        "dC_dv_fd_h_half": float(dvd_half),
        "dC_dv_abs_diff": float(abs(dvd_h - dvd_half)),
    }

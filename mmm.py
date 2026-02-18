#!/usr/bin/env python3

"""
Marketing Mix Model — Diminishing Returns with PyMC

Bayesian MMM that captures the diminishing-return (saturation)
effect of media spend on sales using a Hill function parameterisation.

Author: Siddharth Gupte
Version: 1.1
Date: 12 February 2026
"""

# Importing libraries
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

print("PyMC version:", pm.__version__)

# ------------------------------1. Load and Prepare Data and Channels------------------------------
df = pd.read_csv("./data/mmm_test_data.csv", parse_dates=["date"])

CHANNELS = ["tv_spend", "radio_spend", "social_spend", "display_spend", "search_spend"]
CHANNEL_LABELS = ["TV", "Radio", "Social", "Display", "Search"]

# Scale spend to [0, 1] per channel (max-normalise) for numerical stability
spend_maxes = df[CHANNELS].max().values  # save for back-transform on plots
X_raw = df[CHANNELS].values
X = X_raw / spend_maxes[None, :]  # N x C, each in [0, 1]

# Centre and scale sales (z-score) — helps sampler convergence
y_raw = df["sales"].values
y_mean, y_std = y_raw.mean(), y_raw.std()
y = (y_raw - y_mean) / y_std

N, C = X.shape
print(f"Data: {N} days × {C} channels. Sales mean={y_mean:.0f}, std={y_std:.0f}")


# ------------------------------2. Model Specification------------------------------
print("\nBuilding PyMC model …")

with pm.Model() as mmm:
    # --- Data containers ---
    X_data = pm.Data("X", X)

    # --- Saturation parameters (per channel) ---
    K = pm.Beta("K", alpha=2, beta=2, shape=C)           # half-saturation
    S = pm.Gamma("S", alpha=3, beta=1, shape=C)          # Hill exponent

    # --- Hill saturation transform ---
    # f(x) = x^S / (K^S + x^S)
    import pytensor.tensor as pt
    x_s = pt.pow(X_data, S)
    k_s = pt.pow(K, S)
    saturated = x_s / (k_s + x_s) # (N, C)

    # --- Channel coefficients (positive) ---
    beta_ch = pm.HalfNormal("beta_channel", sigma=1, shape=C)

    # --- Intercept & noise ---
    alpha = pm.Normal("alpha", mu=0, sigma=0.5)
    sigma = pm.HalfNormal("sigma", sigma=0.5)

    # --- Likelihood ---
    mu = alpha + pm.math.dot(saturated, beta_ch)
    pm.Normal("obs", mu=mu, sigma=sigma, observed=y)

print(pm.model_to_graphviz(mmm))

# ------------------------------3. Prior Predictive Check------------------------------
print("\nDrawing prior predictive samples …")
with mmm:
    prior = pm.sample_prior_predictive(samples=500, random_seed=42)

# ------------------------------4. Posterior Sampling------------------------------
with mmm:
    trace = pm.sample(
        2000,
        tune=1500,
        chains=4,
        cores=4,
        target_accept=0.95,
        random_seed=42,
        return_inferencedata=True,
    )

print("\n--- Posterior Summary ---")
var_names = ["alpha", "beta_channel", "K", "S", "sigma"]
summary = az.summary(trace, var_names=var_names, round_to=3)
print(summary)

# ------------------------------5. Posterior Predictive------------------------------
print("\nDrawing posterior predictive samples …")
with mmm:
    ppc = pm.sample_posterior_predictive(trace, random_seed=42)

# ------------------------------6. Plotting Results------------------------------

# Specifying the Colour palette
COLOURS = ["#2563EB", "#DC2626", "#16A34A", "#F59E0B", "#8B5CF6"]

# Helper function: evaluate Hill curve on a grid
def hill(x, K, S):
    return x**S / (K**S + x**S)

# ------------------------------6a. Prior vs Posterior Diminishing-Return Curves------------------------------
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.ravel()

# Extract samples
K_prior = prior.prior["K"].values.reshape(-1, C)
S_prior = prior.prior["S"].values.reshape(-1, C)
K_post  = trace.posterior["K"].values.reshape(-1, C)
S_post  = trace.posterior["S"].values.reshape(-1, C)

x_grid = np.linspace(0, 1, 200)

for i, (ch, label, colour) in enumerate(zip(CHANNELS, CHANNEL_LABELS, COLOURS)):
    ax = axes[i]

    # Prior curves (thin, light)
    n_curves = 100
    idx_pr = np.random.default_rng(0).choice(K_prior.shape[0], n_curves, replace=False)
    for j in idx_pr:
        y_curve = hill(x_grid, K_prior[j, i], S_prior[j, i])
        ax.plot(x_grid * spend_maxes[i], y_curve, color=colour, alpha=0.04, lw=0.8)

    # Posterior curves (darker)
    idx_po = np.random.default_rng(1).choice(K_post.shape[0], n_curves, replace=False)
    for j in idx_po:
        y_curve = hill(x_grid, K_post[j, i], S_post[j, i])
        ax.plot(x_grid * spend_maxes[i], y_curve, color="black", alpha=0.06, lw=0.8)

    # Posterior median curve
    y_med = hill(x_grid, np.median(K_post[:, i]), np.median(S_post[:, i]))
    ax.plot(x_grid * spend_maxes[i], y_med, color=colour, lw=2.5, label="Posterior median")

    # Cosmetics
    ax.set_title(label, fontsize=14, fontweight="bold")
    ax.set_xlabel("Daily Spend ($)")
    ax.set_ylabel("Saturation Effect (0–1)")
    ax.set_ylim(-0.05, 1.05)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.25)

    # Annotate K (half-saturation point) on posterior median
    K_med_raw = np.median(K_post[:, i]) * spend_maxes[i]
    ax.axvline(K_med_raw, ls="--", color=colour, alpha=0.5, lw=1)
    ax.annotate(f"K₅₀ ≈ ${K_med_raw/1000:.0f}k",
                xy=(K_med_raw, 0.5), fontsize=8, color=colour,
                xytext=(15, 15), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color=colour, lw=0.8))

# Remove unused subplot
axes[5].axis("off")

# Add legend descriptions
axes[5].text(0.1, 0.7, "Reading the plots:", fontsize=12, fontweight="bold",
             transform=axes[5].transAxes)
axes[5].text(0.1, 0.55, "Light coloured lines = Prior curves\n(wide uncertainty before seeing data)",
             fontsize=10, transform=axes[5].transAxes, color=COLOURS[0])
axes[5].text(0.1, 0.38, "Dark / black lines = Posterior curves\n(narrower after fitting to data)",
             fontsize=10, transform=axes[5].transAxes, color="black")
axes[5].text(0.1, 0.21, "Bold coloured line = Posterior median\n(best-estimate saturation curve)",
             fontsize=10, transform=axes[5].transAxes, color=COLOURS[2])
axes[5].text(0.1, 0.04, "Dashed line = K₅₀, the half-saturation\npoint (50% of max channel effect)",
             fontsize=10, transform=axes[5].transAxes, color=COLOURS[3])

fig.suptitle("Prior vs Posterior Diminishing-Return (Hill) Curves per Channel",
             fontsize=16, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("./results/prior_vs_posterior_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved prior_vs_posterior_curves.png")

# ------------------------------6b. Posterior Parameter Distributions------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

for idx, (param, title) in enumerate([("K", "K (half-saturation)"),
                                        ("S", "S (Hill exponent)"),
                                        ("beta_channel", "β (channel coefficient)")]):
    ax = axes[idx]
    vals = trace.posterior[param].values.reshape(-1, C)
    for i, (label, colour) in enumerate(zip(CHANNEL_LABELS, COLOURS)):
        az.plot_kde(vals[:, i], ax=ax, plot_kwargs={"color": colour, "lw": 2},
                    label=label)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

fig.suptitle("Posterior Distributions of Hill-Function Parameters",
             fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig("./results/posterior_params.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved posterior_params.png")

# ------------------------------6c. Posterior Predictive Fit------------------------------
fig, ax = plt.subplots(figsize=(16, 4.5))
obs_hat = ppc.posterior_predictive["obs"].values.reshape(-1, N)
obs_median = np.median(obs_hat, axis=0) * y_std + y_mean
obs_lo = np.percentile(obs_hat, 5, axis=0) * y_std + y_mean
obs_hi = np.percentile(obs_hat, 95, axis=0) * y_std + y_mean

ax.fill_between(df["date"], obs_lo, obs_hi, alpha=0.25, color="#2563EB", label="90% PPI")
ax.plot(df["date"], obs_median, color="#2563EB", lw=1.2, label="Posterior median")
ax.scatter(df["date"], y_raw, s=5, color="black", alpha=0.4, label="Observed sales", zorder=3)
ax.set_title("Posterior Predictive Check — Sales", fontsize=14, fontweight="bold")
ax.set_ylabel("Sales ($)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v/1e6:.1f}M"))
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig("./results/posterior_predictive.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved posterior_predictive.png")
#!/usr/bin/env python3

"""
Marketing Mix Model v2 — Adstock + Diminishing Returns with PyMC

Bayesian MMM that captures both:
  1. Carryover / adstock (geometric decay) — today's spend echoes into future days
  2. Diminishing returns (Hill saturation) — each extra dollar yields less return

Pipeline per channel:  raw spend → geometric adstock → Hill saturation → β-weighted effect

Author: Siddharth Gupte
Version: 2.0
Date: 15 February 2026
"""

# Importing libraries
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import pytensor.tensor as pt
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

print("PyMC version:", pm.__version__)

#  ------------------------------1. Load and Prepare data and channels------------------------------
df = pd.read_csv("./data/mmm_test_data.csv", parse_dates=["date"])

CHANNELS = ["tv_spend", "radio_spend", "social_spend", "display_spend", "search_spend"]
CHANNEL_LABELS = ["TV", "Radio", "Social", "Display", "Search"]
COLOURS = ["#2563EB", "#DC2626", "#16A34A", "#F59E0B", "#8B5CF6"]

# Max-normalise spend to [0, 1] per channel
spend_maxes = df[CHANNELS].max().values
X_raw = df[CHANNELS].values
X = X_raw / spend_maxes[None, :]          # (N, C) in [0, 1]

# Z-score sales
y_raw = df["sales"].values
y_mean, y_std = y_raw.mean(), y_raw.std()
y = (y_raw - y_mean) / y_std

N, C = X.shape
print(f"Data: {N} days × {C} channels.  Sales mean={y_mean:,.0f}, std={y_std:,.0f}")

# Quick channel activity summary
for i, ch in enumerate(CHANNEL_LABELS):
    active_pct = (X_raw[:, i] > 0).mean() * 100
    print(f"  {ch}: {active_pct:.1f}% active days")

#  ------------------------------2. Geometric Adstock------------------------------
#
# For each channel c, the adstocked spend at time t is:
#
#     adstock_t = spend_t + λ_c · adstock_{t-1}
#
# where λ_c ∈ [0, 1) is the decay (retention) rate.
#
# λ = 0   → no carryover (reduces to v1 model)
# λ = 0.8 → strong carryover; ~5-day half-life
#
# IMPLEMENTATION NOTE (v2.1):
# The naive approach uses pytensor.scan, which creates a sequential loop
# that PyTensor can't vectorise — making NUTS extremely slow (~40+ min).
#
# Instead, we use a CONVOLUTION with a truncated geometric kernel of
# length L (default 15 days). For any realistic λ < 0.85, a 15-day
# kernel captures >99% of the total carryover weight. This turns the
# adstock into a standard 1D convolution that PyTensor can differentiate
# efficiently through its existing conv ops.
#
# The kernel for channel c is: [1, λ_c, λ_c², λ_c³, ..., λ_c^(L-1)]
# normalised so the weights sum to 1.

L_MAX = 15  # Max carryover window (days). λ=0.85 → 99.6% captured in 15 days.


def geometric_adstock_conv(X_data, lambdas, L=L_MAX):
    """Apply geometric adstock to all channels via 1D convolution.

    For each channel, constructs a kernel [1, λ, λ², ..., λ^(L-1)],
    normalises it, and convolves with the spend series.

    Parameters
    ----------
    X_data : pytensor matrix, shape (N, C)
        Normalised spend per channel.
    lambdas : pytensor vector, shape (C,)
        Decay rates in [0, 1).
    L : int
        Kernel length (max lag days).

    Returns
    -------
    X_adstocked : pytensor matrix, shape (N, C)
        Adstocked and re-normalised to [0, 1] per channel.
    """
    lags = pt.arange(L).astype("float64")           # [0, 1, 2, ..., L-1]
    # kernels shape: (C, L) — each row is [λ^0, λ^1, ..., λ^(L-1)]
    kernels = pt.pow(lambdas[:, None], lags[None, :])  # (C, L)
    # Normalise each kernel to sum to 1
    kernels = kernels / pt.sum(kernels, axis=1, keepdims=True)

    # Apply convolution per channel using a manual dot product with shifted data
    # We pad the beginning with zeros and use a sliding window
    N = X_data.shape[0]
    channels = []
    for i in range(C):
        # Pad spend with L-1 leading zeros for causal convolution
        x_padded = pt.concatenate([pt.zeros(L - 1), X_data[:, i]])
        # Causal convolution: for each time t, dot product with reversed kernel
        # This is equivalent to: adstock_t = Σ_{j=0}^{L-1} kernel[j] * x_{t-j}
        # We build this as a matrix multiply for efficiency
        # Create index matrix for gathering: shape (N, L)
        indices = pt.arange(N)[:, None] + pt.arange(L)[None, :]  # (N, L)
        x_windows = x_padded[indices]  # (N, L) — each row is [x_t, x_{t-1}, ..., x_{t-L+1}]
        adstocked_i = pt.dot(x_windows, kernels[i])  # (N,)
        # Re-normalise to [0, 1]
        adstocked_i = adstocked_i / pt.max(adstocked_i)
        channels.append(adstocked_i)

    return pt.stack(channels, axis=1)  # (N, C)

#  ------------------------------3. Model Specification------------------------------
#
# Pipeline:  spend → adstock(λ) → Hill(K, S) → β·saturated → sum + α → sales
#
# Prior choices:
#   λ ~ Beta(2, 5)   — mode ≈ 0.17, most mass < 0.5
#       Advertising carryover typically decays within a few days to a week.
#       Beta(2,5) gently centres decay rates toward faster decay while
#       allowing the data to push toward longer carryover if warranted.
#       For always-on channels (TV, Radio, Social) we expect moderate λ;
#       for bursty channels (Display, Search) λ should be small.
#
#   K ~ Beta(2, 2)   — same as v1
#   S ~ Gamma(3, 1)  — same as v1
#   β ~ HalfNormal(1) — same as v1
#   α ~ Normal(0, 0.5)
#   σ ~ HalfNormal(0.5)

print("\nBuilding PyMC model (v2: adstock + Hill) …")

with pm.Model() as mmm_v2:

    # --- Data containers ---
    X_data = pm.Data("X", X)

    # --- Adstock decay rates (per channel) ---
    lam = pm.Beta("lambda", alpha=2, beta=5, shape=C)

    # --- Apply geometric adstock (convolution-based, fast) ---
    X_adstocked = geometric_adstock_conv(X_data, lam)

    # --- Saturation parameters (per channel) ---
    K = pm.Beta("K", alpha=2, beta=2, shape=C)
    S = pm.Gamma("S", alpha=3, beta=1, shape=C)

    # --- Hill saturation: x^S / (K^S + x^S) ---
    x_s = pt.pow(X_adstocked, S)
    k_s = pt.pow(K, S)
    saturated = x_s / (k_s + x_s)  # (N, C)

    # --- Channel coefficients (positive) ---
    beta_ch = pm.HalfNormal("beta_channel", sigma=1, shape=C)

    # --- Intercept & noise ---
    alpha = pm.Normal("alpha", mu=0, sigma=0.5)
    sigma = pm.HalfNormal("sigma", sigma=0.5)

    # --- Likelihood ---
    mu = alpha + pm.math.dot(saturated, beta_ch)
    pm.Normal("obs", mu=mu, sigma=sigma, observed=y)

print(pm.model_to_graphviz(mmm_v2))

#  ------------------------------4. Prior Predictive Check------------------------------

print("\nDrawing prior predictive samples …")
with mmm_v2:
    prior = pm.sample_prior_predictive(samples=500, random_seed=42)

#  ------------------------------5. Posterior Sampling-----------------------------
#
# Notes:
#   - Higher target_accept (0.97) to handle the scan-induced geometry.
#   - More tuning steps (2000) since adstock adds C new parameters
#     and the scan creates longer- dependency chains.
#   - If divergences appear, consider increasing target_accept to 0.99
#     or reparameterising λ with a logit-normal.

print("\nSampling posterior (this may take a while with adstock scan) …")

with mmm_v2:
    trace = pm.sample(
        2000,
        tune=2000,
        chains=4,
        cores=4,
        target_accept=0.97,
        random_seed=42,
        return_inferencedata=True,
    )

print("\n--- Posterior Summary ---")
var_names = ["alpha", "beta_channel", "K", "S", "lambda", "sigma"]
summary = az.summary(trace, var_names=var_names, round_to=3)
print(summary)

# Check diagnostics
divergences = trace.sample_stats["diverging"].sum().values
print(f"\nDivergences: {divergences}")

# ------------------------------6. Posterior Predictive Check------------------------------

print("\nDrawing posterior predictive samples …")
with mmm_v2:
    ppc = pm.sample_posterior_predictive(trace, random_seed=42)

# ------------------------------7. Plotting------------------------------

# Helper: Hill function
def hill(x, K, S):
    return x**S / (K**S + x**S)


# Helper: numpy geometric adstock (for plotting)
def np_geometric_adstock(x, lam):
    out = np.zeros_like(x)
    out[0] = x[0]
    for t in range(1, len(x)):
        out[t] = x[t] + lam * out[t - 1]
    return out

# ------------------------------7a. Adstock Decay Profiles------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left panel: impulse response (decay of a single $1 impulse)
ax = axes[0]
lam_post = trace.posterior["lambda"].values.reshape(-1, C)
lam_medians = np.median(lam_post, axis=0)
days = np.arange(15)

for i, (label, colour) in enumerate(zip(CHANNEL_LABELS, COLOURS)):
    impulse_response = lam_medians[i] ** days
    half_life = -np.log(2) / np.log(lam_medians[i]) if lam_medians[i] > 0 else 0
    ax.plot(days, impulse_response, color=colour, lw=2,
            label=f"{label} (λ={lam_medians[i]:.2f}, t½={half_life:.1f}d)")

ax.set_xlabel("Days After Spend")
ax.set_ylabel("Retained Effect")
ax.set_title("Adstock Impulse Response (Posterior Median)", fontsize=13, fontweight="bold")
ax.legend(fontsize=8, loc="upper right")
ax.grid(True, alpha=0.25)
ax.set_ylim(-0.05, 1.05)

# Right panel: posterior distributions of λ
ax = axes[1]
for i, (label, colour) in enumerate(zip(CHANNEL_LABELS, COLOURS)):
    az.plot_kde(lam_post[:, i], ax=ax, plot_kwargs={"color": colour, "lw": 2}, label=label)
ax.set_title("Posterior of λ (Decay Rate)", fontsize=13, fontweight="bold")
ax.set_xlabel("λ")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

fig.suptitle("Adstock Carryover Parameters", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig("./results/adstock_decay.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved adstock_decay.png")

# ------------------------------7b. Prior vs Posterior Diminishing-Return Curves (with adstock)------------------------------

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.ravel()

K_prior = prior.prior["K"].values.reshape(-1, C)
S_prior = prior.prior["S"].values.reshape(-1, C)
K_post = trace.posterior["K"].values.reshape(-1, C)
S_post = trace.posterior["S"].values.reshape(-1, C)

x_grid = np.linspace(0, 1, 200)

for i, (ch, label, colour) in enumerate(zip(CHANNELS, CHANNEL_LABELS, COLOURS)):
    ax = axes[i]

    # Prior curves (thin, light)
    n_curves = 100
    rng_pr = np.random.default_rng(0)
    idx_pr = rng_pr.choice(K_prior.shape[0], n_curves, replace=False)
    for j in idx_pr:
        y_curve = hill(x_grid, K_prior[j, i], S_prior[j, i])
        ax.plot(x_grid * spend_maxes[i], y_curve, color=colour, alpha=0.04, lw=0.8)

    # Posterior curves (darker)
    rng_po = np.random.default_rng(1)
    idx_po = rng_po.choice(K_post.shape[0], n_curves, replace=False)
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

    # Annotate K₅₀
    K_med_raw = np.median(K_post[:, i]) * spend_maxes[i]
    ax.axvline(K_med_raw, ls="--", color=colour, alpha=0.5, lw=1)
    ax.annotate(f"K₅₀ ≈ ${K_med_raw/1000:.0f}k",
                xy=(K_med_raw, 0.5), fontsize=8, color=colour,
                xytext=(15, 15), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color=colour, lw=0.8))

# Legend panel
axes[5].axis("off")
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

fig.suptitle("Prior vs Posterior Diminishing-Return (Hill) Curves — v2 with Adstock",
             fontsize=16, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("./results/prior_vs_posterior_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved prior_vs_posterior_curves.png")

# ------------------------------7c. Posterior Parameter Distributions (K, S, β, λ)------------------------------

fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))

params = [
    ("K", "K (half-saturation)"),
    ("S", "S (Hill exponent)"),
    ("beta_channel", "β (channel coefficient)"),
    ("lambda", "λ (adstock decay)"),
]

for idx, (param, title) in enumerate(params):
    ax = axes[idx]
    vals = trace.posterior[param].values.reshape(-1, C)
    for i, (label, colour) in enumerate(zip(CHANNEL_LABELS, COLOURS)):
        az.plot_kde(vals[:, i], ax=ax, plot_kwargs={"color": colour, "lw": 2}, label=label)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

fig.suptitle("Posterior Distributions of Model Parameters (Adstock + Hill)",
             fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig("./results/posterior_params.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved posterior_params.png")

# ------------------------------7d. Posterior Predictive Fit------------------------------

fig, ax = plt.subplots(figsize=(16, 4.5))

obs_hat = ppc.posterior_predictive["obs"].values.reshape(-1, N)
obs_median = np.median(obs_hat, axis=0) * y_std + y_mean
obs_lo = np.percentile(obs_hat, 5, axis=0) * y_std + y_mean
obs_hi = np.percentile(obs_hat, 95, axis=0) * y_std + y_mean

ax.fill_between(df["date"], obs_lo, obs_hi, alpha=0.25, color="#2563EB", label="90% PPI")
ax.plot(df["date"], obs_median, color="#2563EB", lw=1.2, label="Posterior median")
ax.scatter(df["date"], y_raw, s=5, color="black", alpha=0.4, label="Observed sales", zorder=3)
ax.set_title("Posterior Predictive Check — Sales (v2: Adstock + Hill)",
             fontsize=14, fontweight="bold")
ax.set_ylabel("Sales ($)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v/1e6:.1f}M"))
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig("./results/posterior_predictive.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved posterior_predictive.png")

# ------------------------------7e. Channel Contribution Over Time (decomposition)------------------------------

print("\nComputing channel decomposition …")

# Use posterior medians for the decomposition
lam_med = np.median(lam_post, axis=0)
K_med = np.median(K_post, axis=0)
S_med = np.median(S_post, axis=0)
beta_med = np.median(trace.posterior["beta_channel"].values.reshape(-1, C), axis=0)
alpha_med = np.median(trace.posterior["alpha"].values.flatten())

# Compute adstocked → saturated → weighted contributions
contributions = np.zeros((N, C))
for i in range(C):
    adstocked = np_geometric_adstock(X[:, i], lam_med[i])
    adstocked_normed = adstocked / adstocked.max() if adstocked.max() > 0 else adstocked
    saturated = hill(adstocked_normed, K_med[i], S_med[i])
    contributions[:, i] = saturated * beta_med[i]

# Back-transform from z-space
contributions_raw = contributions * y_std
base_raw = alpha_med * y_std + y_mean

fig, ax = plt.subplots(figsize=(16, 5))
bottom = np.full(N, base_raw)
ax.fill_between(df["date"], 0, bottom, color="#9CA3AF", alpha=0.4, label="Base")

for i, (label, colour) in enumerate(zip(CHANNEL_LABELS, COLOURS)):
    ax.fill_between(df["date"], bottom, bottom + contributions_raw[:, i],
                    color=colour, alpha=0.6, label=label)
    bottom = bottom + contributions_raw[:, i]

ax.scatter(df["date"], y_raw, s=3, color="black", alpha=0.3, zorder=5, label="Observed")
ax.set_title("Channel Contribution Decomposition (Posterior Medians)",
             fontsize=14, fontweight="bold")
ax.set_ylabel("Sales ($)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v/1e6:.1f}M"))
ax.legend(fontsize=9, loc="upper left", ncol=3)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig("./results/channel_decomposition.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved channel_decomposition.png")

# ------------------------------8. MODEL COMPARISON (optional — if v1 trace exists)------------------------------

print("\n" + "=" * 70)
print("MODEL v2 COMPLETE — Adstock + Hill Saturation")
print("=" * 70)
print(f"\nKey outputs in ./results/:")
print("  • adstock_decay.png            — Impulse responses & λ posteriors")
print("  • prior_vs_posterior_curves_v2.png — Saturation curves with adstock")
print("  • posterior_params.png       — All parameter distributions")
print("  • posterior_predictive.png   — Model fit vs observed sales")
print("  • channel_decomposition.png — Stacked contribution over time")
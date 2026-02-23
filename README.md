# Marketing Mix Model v2 — Adstock + Diminishing Returns with PyMC

## What Changed from v1

Version 1 modelled only **saturation** (diminishing returns via the Hill function). Version 2 adds **geometric adstock** before saturation, creating the standard two-stage media transformation pipeline used in production MMMs:

```
raw spend → adstock(λ) → Hill(K, S) → β·saturated → Σ + α → sales
```

### Why Adstock Before Saturation?

Advertising effects don't vanish the moment you stop spending. A TV ad seen on Monday still influences purchase decisions on Tuesday and Wednesday — this is the **carryover** or **adstock** effect. By applying adstock *before* the Hill function, we model a realistic pipeline:

1. **Adstock** accumulates and decays spend over time → "effective exposure"
2. **Hill saturation** maps effective exposure to a bounded sales response

The ordering matters. Applying saturation first would compress spend into [0, 1] before accumulation, losing the important distinction between "one big day" and "sustained moderate spend over a week." The adstock-then-saturate ordering is the standard in Meta Robyn, Google Meridian, and the academic MMM literature (Jin et al., 2017).

---

## Model Specification

$$\text{sales} = \alpha + \sum_{c} \beta_c \cdot \text{hill}\bigl(\text{adstock}(x_c;\; \lambda_c);\; K_c, S_c\bigr) + \varepsilon$$

**Geometric Adstock:**

$$\text{adstock}_t = x_t + \lambda_c \cdot \text{adstock}_{t-1}$$

where $\lambda_c \in [0, 1)$ is the decay (retention) rate. The half-life of the carryover effect is $t_{1/2} = -\ln 2 / \ln \lambda$.

After adstock, the series is re-normalised to [0, 1] before entering the Hill function.

**Hill Saturation Function:**

$$\text{hill}(x;\; K, S) = \frac{x^S}{K^S + x^S}$$

### Parameters

| Parameter | Description |
|-----------|-------------|
| $\lambda_c$ (decay rate) | Carryover retention per day for channel $c$. λ=0 means no carryover; λ=0.7 means ~2-day half-life |
| $K_c$ (half-saturation) | The effective-exposure level at which channel $c$ reaches 50% of its maximum effect |
| $S_c$ (Hill exponent) | Controls curvature — $S \approx 1$ gives Michaelis–Menten; $S > 1$ creates an S-shape |
| $\beta_c$ (coefficient) | Maximum possible sales lift (in z-scored units) from channel $c$ at full saturation |
| $\alpha$ (intercept) | Baseline sales level when all channels are at zero spend |
| $\sigma$ (noise) | Observation noise standard deviation |

---

## Prior Choices & Justification

| Parameter | Prior | Rationale |
|-----------|-------|-----------|
| $\lambda$ | Beta(2, 5) | Mode ≈ 0.17, mean ≈ 0.29. Most advertising carryover decays within a few days. This prior centres on fast decay while allowing moderate carryover (λ ≈ 0.5–0.7 for TV) if the data supports it. Chosen over Beta(1,1)/Uniform because very high λ values (>0.9) imply month-long half-lives, which are implausible for direct sales response. |
| $K$ | Beta(2, 2) | Same as v1. Symmetric prior on [0, 1], mildly informative. |
| $S$ | Gamma(3, 1) | Same as v1. Centres the Hill exponent around 2–3. |
| $\beta_c$ | HalfNormal(1) | Same as v1. Enforces positivity and mild regularisation. |
| $\alpha$ | Normal(0, 0.5) | Weakly informative in z-space. |
| $\sigma$ | HalfNormal(0.5) | Weakly informative. |

### Note on Identifiability

Adding adstock introduces potential identifiability tension between λ (how much carry-over) and K (where saturation kicks in). A channel with high λ and low K can produce similar fits to one with low λ and high K. The informative priors on both parameters help regularise this, and the diagnostic checks (r̂, ESS, divergences) should be examined carefully. If identifiability issues arise, consider:

1. Fixing λ for one channel as a reference
2. Using stronger priors informed by experimental data (e.g., geo-lift tests)
3. Reparameterising λ via logit-normal for better sampling geometry

---

## Sampling Configuration

| Setting | v1 | v2 | Rationale for change |
|---------|----|----|---------------------|
| Tune | 1,500 | 2,000 | Adstock scan creates longer dependency chains |
| Draws | 2,000 | 2,000 | Unchanged |
| Chains | 4 | 4 | Unchanged |
| target_accept | 0.95 | 0.97 | Higher to handle scan-induced posterior geometry |

---

## Outputs

The script generates five plots in `./results/`:

| File | Description |
|------|-------------|
| `adstock_decay.png` | Impulse response curves and posterior distributions of λ per channel |
| `prior_vs_posterior_curves_v2.png` | Prior vs posterior Hill saturation curves (same as v1 but with adstock-adjusted K values) |
| `posterior_params_v2.png` | KDE plots of all four parameter types: K, S, β, λ |
| `posterior_predictive_v2.png` | Observed sales vs model posterior predictive (90% interval) |
| `channel_decomposition_v2.png` | Stacked area chart showing each channel's contribution to sales over time |

---

## What's Still Missing (v3 Roadmap)

- **Trend & seasonality** — linear/spline trend + Fourier terms or day-of-week effects
- **Control variables** — pricing, promotions, holidays, competitor activity
- **Weibull adstock** — more flexible than geometric; can model delayed peak effects
- **Out-of-sample validation** — time-based train/test split with MAPE and coverage metrics
- **Budget optimiser** — marginal ROI–based reallocation given the fitted model
- **Experimental calibration** — using geo-lift test results as informative priors on β
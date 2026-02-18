# Marketing Mix Model — Diminishing Returns with PyMC

## Model Specification

Daily sales are modelled as a linear combination of saturated media channel effects:

$$\text{sales} = \alpha + \sum_{c} \beta_c \cdot \text{hill}(x_c;\; K_c, S_c) + \varepsilon$$

where the **Hill saturation function** is:

$$\text{hill}(x;\; K, S) = \frac{x^S}{K^S + x^S}$$

applied to max-normalised daily spend per channel. This is a standard dose–response curve borrowed from pharmacokinetics and widely used in MMM frameworks (Meta Robyn, Google Meridian). It maps any spend level to a value in [0, 1], naturally capturing the economic intuition that each additional dollar of spend yields less incremental return.

### Why Hill?

- **Bounded in [0, 1]** — prevents extrapolation blow-up, unlike power-law ($x^\alpha$) which is unbounded.
- **Interpretable parameters** — $K$ tells you "how fast does this channel saturate?" and $S$ tells you "how sharp is the curve?"
- **Industry standard** — default in Meta's Robyn and Google's Meridian / LightweightMMM.
- A power-law $x^\alpha$ is unbounded and can overfit high spenders; $\log(1+x)$ lacks a natural ceiling.

### Parameters

| Parameter | Description |
|-----------|-------------|
| $K_c$ (half-saturation) | The normalised spend level at which channel $c$ reaches 50% of its maximum effect |
| $S_c$ (Hill exponent) | Controls curvature — $S \approx 1$ gives Michaelis–Menten; $S > 1$ creates an S-shape |
| $\beta_c$ (coefficient) | Maximum possible sales lift (in z-scored units) from channel $c$ at full saturation |
| $\alpha$ (intercept) | Baseline sales level when all channels are at zero spend |
| $\sigma$ (noise) | Observation noise standard deviation |

---

## Prior Choices & Justification

| Parameter | Prior | Rationale |
|-----------|-------|-----------|
| $K$ | Beta(2, 2) | Symmetric prior on [0, 1], mildly informative. Centres the half-saturation point at 50% of observed max spend but has enough variance to let data push $K$ toward 0 (fast saturation) or 1 (slow saturation). Chosen over Uniform to down-weight extreme values ($K \approx 0$ or $1$) which would imply instant or never-reached saturation. |
| $S$ | Gamma(3, 1) | Centres the Hill exponent around 2–3 (mode = 2). Gently regularises toward smooth concave/sigmoid curves. Extremely large $S$ creates a near-step-function, which is economically implausible for advertising; the Gamma tail penalises that without ruling it out. |
| $\beta_c$ | HalfNormal(1) | Enforces the sign constraint (more spend should not decrease sales) and places most prior mass on 0–2 z-scored standard deviations of sales contribution per channel — reasonable for five channels sharing a total effect. |
| $\alpha$ | Normal(0, 0.5) | Weakly informative in z-space; centred at the mean by construction of the z-score transform. |
| $\sigma$ | HalfNormal(0.5) | Weakly informative; we expect a decent fit so residual std should be well below 1 z-unit. |

---

## Sampling Diagnostics

The model was sampled with 4 NUTS chains (1,500 tuning + 2,000 draws each, `target_accept=0.95`). Key diagnostics:

- **0 divergences** across all chains
- **r̂ = 1.000** for all parameters (perfect convergence)
- **Effective sample sizes (ESS)** ranging from ~3,000 to ~10,000 (well above recommended minimums)

---

## What the Prior vs Posterior Curves Show

The **prior curves** (light, spread-out spaghetti lines) show the wide range of saturation behaviours the model considers plausible *before* seeing any data. They span nearly the full [0, 1] range of shapes — from nearly linear to sharply saturating.

The **posterior curves** (dark/bold lines) collapse into a much narrower band after fitting, showing what the data actually supports. For each channel:

- The **steepness** tells you how quickly returns diminish.
- The **K₅₀ dashed line** marks the spend level at which the channel reaches half its maximum effect — a key budget planning insight.
- Channels where the posterior band is still wide (e.g., Social) have more uncertainty in their saturation shape, while channels with tight posteriors (e.g., Display, Search) are well-identified by the data.

### Key Posterior Findings

| Channel | K₅₀ (half-saturation spend) | S (shape) | β (max effect) | Interpretation |
|---------|------------------------------|-----------|-----------------|----------------|
| TV | ~$52k/day | ~4.7 (steep S-curve) | 0.95 | Moderate effect, saturates at higher spend |
| Radio | ~$14k/day | ~1.7 (gentle concave) | 1.30 | Strong steady contribution, gradual saturation |
| Social | ~$5k/day | ~2.3 (moderate) | 0.42 | Smaller effect, saturates relatively early |
| Display | ~$39k/day | ~3.8 (steep) | 2.25 | Large but quickly-saturating effect on active days |
| Search | ~$94k/day | ~3.4 (steep) | 4.65 | Largest per-activation effect, high saturation ceiling |

Display and Search are bursty channels (active on only ~8% and ~3% of days respectively), so their high β values reflect large sales lifts on the days they fire, with tight posterior uncertainty driven by the sharp contrast between on/off days.

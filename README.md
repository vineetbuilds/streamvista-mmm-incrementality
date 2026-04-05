# Subscriber Acquisition Incrementality Measurement
## Bayesian Marketing Mix Modeling (MMM) — Pay-TV & Streaming

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![PyMC](https://img.shields.io/badge/PyMC-5.0+-orange) ![License](https://img.shields.io/badge/license-MIT-green)

---

## TL;DR

**The business problem:** A subscription streaming platform sees new subscribers increase from ~4,000 to ~5,000 month-over-month — a +25% jump. A promotional campaign (30% discount on first 3 months) was running this month but not last month. The platform has no individual customer data — no names, no click history, no demographics — just a single aggregate number from Google Analytics each week. The question is deceptively simple: *of that +1,000 subscriber increase, how many were actually caused by the promotion, and how many would have joined anyway?*

**Why this is hard:** Multiple things changed simultaneously. The promotion launched. NFL season started. No competitor ran a promo this month. Consumer confidence shifted. The platform was already on a slow organic upward trend. If you just look at the before/after numbers and give the promotion full credit, you're almost certainly wrong — and you'd be making future budget decisions based on a badly inflated sense of what the campaign actually did.

**Why not attribution modeling:** Attribution modeling assigns credit to individual marketing touchpoints and requires person-level data — each customer's clicks, ad exposures, and the moment they converted. That data doesn't exist here. All we have are weekly aggregate totals. **Marketing Mix Modeling (MMM)** is the correct methodology: it works entirely at the aggregate level, decomposing total subscriber volume into contributions from each driver using time-series regression.

**What we built:** A full end-to-end **Bayesian Marketing Mix Model** in Python using PyMC. The pipeline covers: synthetic data generation (104 weeks of weekly data across 10 variables), **adstock transformation** to capture advertising carryover effects (TV ads from 3 weeks ago still influence this week's subscribers), **log transformation** for diminishing returns (doubling spend doesn't double subscribers), a **Bayesian linear model** with domain-informed priors, NUTS sampling for full posterior distributions, convergence diagnostics (R-hat, ESS), **counterfactual-based incrementality** estimation, ROI calculation, and a six-panel analytical dashboard.

**Key modeling decisions:**
- *Adstock decay rates:* TV = 0.65 (slow decay, brand awareness lingers weeks), Digital = 0.30 (fast decay, paid search is near-immediate), Email = 0.20 (very fast decay)
- *Priors:* Marketing channels get HalfNormal priors — effect must be ≥ 0 because advertising cannot make subscribers go down. Competitor promotions get a Normal prior centered negative — expected to suppress acquisition. Macro index gets a symmetric Normal — effect direction genuinely ambiguous
- *Why Bayesian over regular regression:* Instead of "the promo added 796 subscribers" (false precision), we get "we are 90% confident the promo added between 685 and 1,103 subscribers" — honest uncertainty that changes how you think about investment risk
- *Counterfactual:* We rerun the model with promo contribution zeroed out during the campaign window. The gap between actual and counterfactual = incremental subscribers

**Inputs to the model:**
Marketing-controlled: TV spend, digital/search spend, email volume, promo flag, promo discount depth.
External confounders: competitor promotion activity, sports season flag (NFL), annual seasonality, consumer confidence index, organic trend.

**What the model found:** Of the observed subscriber increase during the campaign window, approximately **6% is attributable to the promotion** (incremental) and **94% is organic baseline and external factors** — primarily seasonality, NFL season overlap, and underlying trend. The 90% credible interval on incremental subscribers is **[685 — 1,103]**. This means a naive before/after comparison would overstate the promotion's impact by roughly 16×.

**On the ROI:** The model also estimates a negative ROI (−24%) on the campaign — which is not a contradiction of the 6% lift. These measure different things. The 6% lift confirms the campaign genuinely caused 774 incremental subscribers. The negative ROI reflects that at $50/month ARPU over an 18-month LTV assumption, those 774 subscribers generate ~$696k in lifetime revenue against ~$913k in campaign spend. The short LTV horizon is the primary driver — at 36 months, the ROI flips positive. This is a realistic and important nuance that MMM surfaces: a campaign can be effective (it moved subscribers) while also being unprofitable at a given LTV assumption. Knowing which it is changes the decision entirely.

**The dashboard:** Six panels — (1) observed vs. model fitted volume over 2 years with promo and competitor windows annotated, (2) stacked area decomposition showing what drove each week's subscriber count, (3) incrementality close-up showing actual vs. counterfactual during the campaign window with the incremental gap shaded green, (4) posterior forest plot showing credible intervals for every coefficient, (5) 2-year attribution pie chart showing overall channel share, (6) scenario analysis projecting subscriber volume under alternative campaign configurations.

**What a production version would add:** Learned adstock decay rates (as Bayesian parameters rather than hardcoded), Hill function saturation curves (replacing log transform), geo-level hierarchical model across DMAs, holdout validation with MAPE reporting, geo-based experiments to validate MMM estimates externally, budget optimization layer, and automated pipeline via Databricks and Snowflake.

---

## About This Project

Incrementality measurement is one of the hardest and most consequential problems in subscription marketing analytics. When aggregate subscriber counts change month-over-month, how much of that change is genuinely *caused* by marketing activity versus organic growth, seasonality, competitor behavior, or macroeconomic shifts? This project builds a complete, end-to-end Bayesian Marketing Mix Model (MMM) to answer exactly that question for a simulated pay-TV and streaming subscription business.

The methodology — Bayesian linear MMM with geometric adstock, log-linear diminishing returns, and counterfactual incrementality — reflects current industry practice at companies running large-scale subscription acquisition programs.

---

## Skills Demonstrated

| Area | Specifics |
|---|---|
| **Bayesian inference** | PyMC model construction, NUTS sampling, prior specification, posterior diagnostics |
| **Marketing analytics** | MMM methodology, adstock transformation, incrementality measurement, counterfactual analysis |
| **Causal reasoning** | Confounder isolation, counterfactual framework, external factor decomposition |
| **Statistical modeling** | Credible intervals, R-hat convergence, ESS, HalfNormal / Normal prior design |
| **Data engineering** | Synthetic data simulation, feature normalization, time-series structuring |
| **Visualization** | Six-panel analytical dashboard, stacked decomposition, forest plots, scenario analysis |
| **Communication** | Executive-ready findings, production roadmap, assumption documentation |

---

## Business Problem

A streaming subscription platform observes the following in its web analytics dashboard:

| Period | New Subscribers |
|---|---|
| Last Month (T-1) | ~4,000 |
| This Month (T) | ~5,000 |
| Net Change | **+1,000 (+25%)** |

A promotional campaign — 30% discount on the first three months of service — was active this month but was not active last month.

**The question:** Of the +1,000 subscriber increase, how many were *caused* by the promotion (incremental), and how many would have joined anyway (organic baseline)?

This is not a trivial question. Multiple factors changed simultaneously during the measurement window:

- A promotional campaign launched (our variable of interest)
- NFL season was underway, driving higher organic interest in live TV
- No major competitor ran a promotion during this window
- Consumer confidence shifted modestly
- The platform's subscriber base was already on a slow organic upward trend

Without isolating these confounders, naively crediting all subscriber growth to the campaign produces a severely inflated — and misleading — estimate of marketing effectiveness.

---

## Why Marketing Mix Modeling — Not Attribution Modeling

This distinction is worth stating explicitly, because the two terms are frequently conflated.

**Attribution modeling** assigns conversion credit to individual marketing touchpoints. It requires granular, person-level data: each customer's ad exposures, clicks, page visits, and the moment of conversion. Attribution is the right tool when you can track individual customer journeys.

**Marketing Mix Modeling (MMM)** operates entirely at the aggregate level. It requires only time-series totals: weekly subscriber counts, weekly spend per channel, and contextual signals. No individual-level data is needed or used.

Since this analysis works with aggregate web analytics counts — with no individual customer identifiers, clickstream data, or attribution signals — **MMM is the correct and only viable methodology**. Attribution modeling cannot be applied here.

MMM decomposes total subscriber volume into additive contributions from each driver:

```
Subscribers(t) = Organic Baseline(t)
               + TV Advertising Effect(t)
               + Digital Advertising Effect(t)
               + Email Effect(t)
               + Promotion Effect(t)
               + Competitor Activity Effect(t)
               + Sports Season Effect(t)
               + Macroeconomic Effect(t)
               + Noise(t)
```

---

## Data & Inputs

All data in this implementation is synthetically simulated to mirror realistic patterns. In production, each input would be sourced from the systems described below.

### Marketing Inputs (variables the business controls)

| Variable | Description | Production Source |
|---|---|---|
| `tv_spend` | Weekly TV advertising spend ($) | Media planning system (e.g., Mediaocean) |
| `digital_spend` | Weekly paid search / digital spend ($) | Google Ads, DSP platforms |
| `email_volume` | Weekly outbound email campaign volume | ESP (e.g., Salesforce Marketing Cloud) |
| `promo_flag` | Binary: is a promotional campaign active this week? | Internal promotions calendar |
| `promo_depth` | Discount depth when active (e.g., 0.30 = 30% off) | Internal promotions calendar |

### External Confounders (variables the business does NOT control)

| Variable | Description | Production Source |
|---|---|---|
| `competitor_promo` | Binary: did a major competitor run a promotion? | SimilarWeb, Nielsen, competitive intelligence |
| `sports_flag` | Binary: is a major sports season active? | Public NFL/NBA/MLB schedule |
| `seasonality` | Cyclical annual pattern in subscriber demand | Derived from historical data |
| `macro_index` | Consumer confidence proxy, normalized 0–1 | Conference Board, BLS |
| `trend` | Underlying organic growth trajectory | Derived from historical subscriber data |

### Target Variable

| Variable | Description | Production Source |
|---|---|---|
| `subscribers` | Weekly new subscriber count | Web analytics platform (e.g., Google Analytics) |

---

## Methodology

### Step 1 — Adstock Transformation

Advertising does not only affect consumer behavior in the week it runs. A television commercial seen in week 1 may still influence a subscriber's decision in weeks 2, 3, or 4. This carryover effect is modeled via **geometric adstock decay**:

```
adstock[t] = spend[t] + decay_rate × adstock[t-1]
```

Decay rates are set by channel based on the expected speed of consumer response:

| Channel | Decay Rate | Rationale |
|---|---|---|
| TV | 0.65 | Slow decay — brand awareness is durable, effects linger weeks |
| Digital / Search | 0.30 | Fast decay — paid search activates near-immediately |
| Email | 0.20 | Very fast decay — email response is near-instantaneous |

Without adstock, a model would severely underestimate TV's true contribution by ignoring its long-lasting influence on brand awareness and consideration.

### Step 2 — Diminishing Returns (Log Transform)

The relationship between advertising spend and subscriber response is non-linear. Doubling a TV budget does not double subscriber acquisition — returns diminish at higher spend levels. A `log1p()` transform applied to adstocked spend captures this empirically observed curve:

```python
tv_log = log1p(tv_adstocked / 1000)
```

*A natural extension for production: replace the log transform with a Hill (S-curve) saturation function with two learnable parameters — saturation point and shape — for more precise characterization of the diminishing returns curve.*

### Step 3 — Feature Normalization

All continuous features are Z-score standardized (mean=0, std=1) before modeling. This is required for numerical stability in the NUTS sampler: variables on very different scales — TV spend in hundreds of thousands of dollars alongside binary 0/1 flags — create poorly conditioned posterior geometries that slow convergence.

### Step 4 — Bayesian Linear MMM

The model is built and sampled in **PyMC** using the **NUTS (No-U-Turn) sampler** — the current state of the art for continuous parameter spaces.

#### Model structure

```
μ(t) = intercept
     + β_trend      × time_index(t)
     + β_season     × seasonality(t)
     + β_tv         × log_adstock_tv(t)
     + β_digital    × log_adstock_digital(t)
     + β_email      × log_adstock_email(t)
     + β_promo      × promo_depth(t)
     + β_competitor × competitor_promo(t)
     + β_sports     × sports_flag(t)
     + β_macro      × macro_index(t)

subscribers(t) ~ Normal(μ(t), σ)
```

#### Prior specification

Rather than fitting single point estimates, the Bayesian approach produces a **full posterior probability distribution** over all plausible coefficient values. This delivers two practical advantages:

1. **Honest uncertainty quantification**: instead of "the promo added 796 subscribers," we report "we are 90% confident the promo added between 685 and 1,103 subscribers." This range is essential for risk-adjusted investment decisions.

2. **Domain knowledge injection via priors**: business logic is encoded directly into the model before data is seen.

| Parameter | Prior | Rationale |
|---|---|---|
| `intercept` | Normal(1000, 300) | Expected organic weekly subscriber baseline |
| `β_tv`, `β_digital`, `β_email` | HalfNormal | Marketing spend can only help, not hurt — effect must be ≥ 0 |
| `β_promo` | HalfNormal(500) | Promotions expected to increase subscriptions |
| `β_sports` | HalfNormal(100) | Sports season expected to increase subscriptions |
| `β_competitor` | Normal(−100, 80) | Competitor promos expected to suppress volume |
| `β_macro` | Normal(0, 100) | Effect direction ambiguous — symmetric prior, data decides |
| `σ` | HalfNormal(100) | Unexplained observation noise — must be positive |

#### Sampler configuration

```
Chains:        2
Draws:         1,000 per chain  (2,000 total posterior samples)
Tuning steps:  1,000 per chain
target_accept: 0.90
```

### Step 5 — Convergence Diagnostics

The model reports **R-hat** (Gelman-Rubin convergence statistic) and **ESS** (Effective Sample Size) for every parameter:

- R-hat < 1.01 → chains have converged to the same posterior (all parameters confirmed)
- ESS > 400 per chain → reliable posterior estimates

### Step 6 — Incrementality via Counterfactual

With the model fitted, incrementality is computed by comparing the actual campaign window outcome against a **counterfactual** — what the model predicts subscriber volume would have been with the promotion removed:

```
counterfactual[t] = fitted[t] - promo_contribution[t]   (campaign window only)

Incremental subscribers = Σ actual[t] - Σ counterfactual[t]
```

This counterfactual implicitly controls for all other factors active during the window — NFL season, macro conditions, organic trend — because the model has already estimated and separated their individual contributions.

A **90% credible interval** on the incrementality estimate is computed by propagating uncertainty through all 2,000 posterior samples of `β_promo`, rather than relying on a single point estimate.

---

## Key Findings

*(Based on synthetic simulation — results illustrative of model behavior)*

| Metric | Value |
|---|---|
| Observed subscribers (campaign window, 8 weeks) | ~12,400 |
| Counterfactual (no promotion) | ~11,620 |
| Incremental subscribers | ~774 |
| Incrementality % | ~6% of total |
| Organic baseline % | ~94% of total |
| 90% Credible Interval | [685 — 1,103] subscribers |

**Key model insights:**

- The majority of observed subscriber growth during the campaign window is attributable to organic baseline, seasonal patterns, and the overlapping NFL season — not the promotion. This is a critical finding: naively attributing all growth to the campaign would overstate marketing effectiveness by approximately 16×.
- Competitor promotional activity measurably suppresses acquisition: approximately −90 subscribers per week when a competitor is running a promotion.
- Sports season (NFL) independently contributes approximately +92 subscribers per week — an external demand signal that must be controlled for to isolate promotion effects.
- TV advertising exhibits an adstock decay of 0.65, meaning brand awareness effects persist for approximately 4–5 weeks after the spend week. This has direct implications for timing: cutting TV spend immediately before a campaign launch erodes the awareness foundation that makes promotions more effective.

**A note on the ROI figure:**

The model estimates a negative ROI (−24%) on this campaign, which at first glance appears to contradict the positive incrementality finding. These are measuring two different things and both are correct.

The 6% incrementality confirms the campaign genuinely caused subscriber growth — 774 incremental subscribers who would not have joined without the promotion. The negative ROI reflects whether those subscribers generated enough lifetime revenue to cover the cost of acquiring them. At $50/month ARPU over an 18-month LTV horizon, those 774 subscribers produce approximately $696k in revenue against $913k in campaign spend.

The 18-month LTV assumption is the primary driver of the negative result. Pay-TV and streaming subscribers typically retain well beyond 18 months — at a 36-month LTV horizon the incremental revenue nearly doubles to ~$1.4M, flipping the ROI firmly positive. Additionally, the simulated campaign spend is quite high relative to the incremental subscriber count ($1,180 per incremental subscriber acquired), which reflects a broader MMM insight: when most subscriber growth is organic, the campaign's cost-per-incremental-acquisition is much higher than a naive cost-per-acquisition calculation would suggest.

In production, ARPU and churn-adjusted LTV would be sourced directly from billing systems, and the ROI calculation would be run across a range of LTV scenarios rather than a single fixed assumption. The model's value here is not the specific ROI number — it's the framework for connecting campaign spend to truly incremental revenue rather than total revenue.

---

## Model Diagnostics

All parameters converge cleanly:

- **R-hat**: 1.000 across all parameters (well below the 1.01 threshold)
- **ESS**: 870–1,870 effective samples (well above the 400 minimum)
- **Divergences**: 0 in both chains

---

## Dashboard

The six-panel dashboard (`streamvista_mmm_dashboard.png`) provides a complete visual narrative of the analysis from raw data through to actionable findings.

---

**Panel 1 — Observed vs. Model Fitted Subscriber Volume**

Shows the full 2-year weekly subscriber time series (blue line) against the model's fitted values (orange line). Green shaded regions mark weeks when a StreamVista promotion was active. Red shaded regions mark weeks when a competitor ran a promotion. Arrows annotate the current campaign window and competitor events directly on the chart.

*Takeaway:* The closer the orange line tracks the blue line, the better the model fit. A well-fitted model is a prerequisite for trusting the decomposition and incrementality estimates in subsequent panels. The shading immediately shows how promotional periods — ours and competitors' — correspond to subscriber volume movements.

---

**Panel 2 — Subscriber Volume Decomposition by Driver**

A stacked area chart where each colored band represents one driver's weekly contribution to subscriber volume. From bottom to top: organic baseline (navy), seasonality (blue), TV advertising (light blue), digital/search (orange), email (yellow), promotion (green), sports season (purple), macro (gray). The black dashed line overlays actual observed counts.

*Takeaway:* This is the most important panel. It answers "what is actually driving our subscriber numbers each week?" The dominant navy band at the bottom shows that most subscriber volume is organic — the platform would acquire most subscribers even with no marketing. The green promotion band visibly grows during the three promo windows. The purple sports band spikes during NFL periods. This decomposition is what makes the model actionable for budget planning: you can see at a glance which channels are contributing meaningfully vs. which are nearly invisible in the stack.

---

**Panel 3 — Incrementality Close-Up (Campaign Window)**

Zooms into the 8-week campaign measurement window. Actual observed counts are shown as dots on a solid line. The dashed blue line is the counterfactual — what the model predicts subscribers would have been if the promotion had never run. The green shaded area between the two lines is the incremental lift. The blue area below the counterfactual is the organic baseline.

*Takeaway:* This is the direct answer to the business question. The green gap is relatively thin compared to the total bar height — visually confirming that ~94% of subscriber volume during this window is organic, with ~6% attributable to the promotion. This panel is what you'd show a CMO asking "did our campaign work?" The credible interval reflects genuine uncertainty — not false precision.

---

**Panel 4 — Posterior Coefficient Estimates (Forest Plot)**

A horizontal bar chart showing the posterior distribution for each model coefficient. The dot marks the posterior median. The bar spans the 90% credible interval. A vertical black line marks zero. Variables whose bar does not cross zero have statistically meaningful effects at the 90% confidence level.

*Takeaway:* Competitor promotions (red, clearly left of zero) measurably suppress acquisition. Promotions and email (green and yellow, right of zero) have meaningful positive effects. TV and digital have positive medians but wider intervals — the model is less certain of their precise magnitude, partly because adstock distributes their effect across multiple weeks. This panel is what distinguishes a Bayesian model from regular regression: you see full uncertainty, not just a point estimate. Bars that cross zero are a signal to not over-invest in a channel based on noisy evidence.

---

**Panel 5 — 2-Year Subscriber Attribution Pie Chart**

Shows the percentage of total fitted subscriber volume across the full 2-year period attributable to each driver.

*Takeaway:* The dominant share belongs to organic baseline — a common and important finding in MMM. It means that marketing efficiency (incremental subscribers per dollar spent) matters more than raw volume. Promotions, despite being an active lever, account for a smaller share than seasonality or sports season — reinforcing that external factors are powerful confounders that must always be controlled for. Without the model, you'd have no way to know this.

---

**Panel 6 — Scenario Analysis**

Projects total subscriber volume under five alternative scenarios during the campaign window: no campaign (baseline only), actual (30% promo), deeper discount (45%), double digital spend, and campaign with peak NFL season overlap. The dashed vertical line marks the no-campaign baseline.

*Takeaway:* This turns the model from a backward-looking measurement tool into a forward-looking planning tool. The most important finding is how relatively small the difference between scenarios is versus the organic baseline — marketing is incrementally valuable but not the primary volume driver. This panel directly informs future decisions: how deep should the next discount be? Is it worth doubling digital spend? Does timing the campaign to coincide with NFL season meaningfully improve outcomes?

---

## Assumptions

- **Linear additive structure**: each driver contributes independently. Channel interaction effects (e.g., TV × digital synergy) are not modeled.
- **Stationary coefficients**: channel effectiveness is assumed constant across the 2-year window. A production model would allow coefficients to drift over time.
- **Hardcoded adstock decay rates**: set by domain knowledge rather than learned from data. Learnable decay rates are a natural Bayesian extension.
- **Log-linear diminishing returns**: approximated via `log1p`. A Hill function with fitted parameters would be more precise at high spend levels.
- **ARPU and LTV figures** used in ROI calculation are illustrative placeholders. The negative ROI estimate reflects the short 18-month LTV horizon assumed. At higher LTV or lower churn assumptions, the ROI improves materially — and even at current assumptions, subscriber acquisition at negative short-term ROI is common in subscription growth strategies where long-term retention is the primary value driver.
- **Synthetic data**: all inputs are simulated. Ground-truth coefficients are known, enabling synthetic validation of model recovery.

---

## Limitations & Production Roadmap

This implementation uses synthetic data for validation purposes. The path from here to production:

1. **Learned adstock decay rates** — make decay rate a Bayesian parameter with a Beta(2,2) prior; let the sampler determine the best value from data
2. **Hill function saturation curves** — replace log transform with a proper S-curve function with learnable saturation and shape parameters
3. **Geo-level hierarchical model** — build a pooled hierarchical model across DMAs / regional markets rather than a single national model
4. **Time-varying coefficients** — allow channel effectiveness to drift over time via Gaussian random walk; captures long-term shifts in media efficiency
5. **Holdout validation with MAPE** — train on 80 weeks, evaluate on held-out 24 weeks; report Mean Absolute Percentage Error on unseen data
6. **Geo-based holdout experiments** — run campaigns in test markets, withhold in control markets; compare MMM estimates against experimental ground truth
7. **Budget optimization layer** — use model response functions to solve the constrained allocation problem: given a fixed total budget, find the channel mix that maximizes incremental subscribers
8. **Automated pipeline** — Databricks orchestration with Snowflake as the data warehouse; weekly automated model refresh and dashboard update

---

## How to Run

### Requirements

```bash
pip install -r requirements.txt
```

### Execute

```bash
python streamvista_mmm_incrementality.py
```

The script will:
1. Simulate 104 weeks of synthetic data
2. Apply adstock transformations and feature normalization
3. Build and sample the Bayesian MMM via PyMC (~30–60 seconds)
4. Print convergence diagnostics and posterior summary
5. Compute incrementality estimates with credible intervals
6. Generate the six-panel dashboard PNG
7. Print the executive summary

---

## Repository Structure

```
streamvista-mmm-incrementality/
│
├── streamvista_mmm_incrementality.py   # Full modeling pipeline (heavily annotated)
├── streamvista_mmm_dashboard.png       # Six-panel analytical dashboard
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Excludes __pycache__, .pyc, .DS_Store
└── README.md                           # This document
```

---

## Dependencies

| Library | Version | Purpose |
|---|---|---|
| `pymc` | ≥ 5.0 | Bayesian modeling and NUTS sampling |
| `arviz` | ≥ 0.15 | Posterior diagnostics, R-hat, ESS, HDI |
| `numpy` | ≥ 1.24 | Numerical computation |
| `pandas` | ≥ 2.0 | Tabular data management |
| `matplotlib` | ≥ 3.7 | Dashboard visualization |
| `scipy` | ≥ 1.10 | Statistical utilities |

---

## License

MIT — free to use, adapt, and build on.

---

*All data is synthetically simulated to enable validation against known ground-truth coefficients. The modeling pipeline, methodology, and production roadmap reflect current industry practices in marketing analytics for subscription businesses at scale.*

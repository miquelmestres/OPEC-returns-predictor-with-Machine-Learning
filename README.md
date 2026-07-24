# OPEC's Returns: Predicting OPEC-Linked Equity Returns with Tree Ensembles and Geopolitical Risk


This is part of an initial trilogy exploring tail shocks and non-linear effects by geopolitics covering ML methods, quantitative finance, and geopolitics rationale built through a Nassim Taleb-influenced lens: average-case metrics are a weak (sometimes actively misleading) description of systems dominated by rare, fat-tailed shocks.

---

## Origin

This specific project extends a previous university group assignment ("OPEC's Returns") that ran a CAPM and a multifactor OLS regression on **monthly** excess returns of seven OPEC-linked oil companies using 5 variables. That original version found a low overall R², with only Brent crude and a hand-built binary "war declared" dummy reaching statistical significance (which was mainly a count of regionally-recognized number of conflicts arising in a specific country). This result was consistent with the idea that these returns are dominated by irregular shocks an OLS coefficient can't represent, rather than by a stable linear relationship to macro fundamentals.

This version keeps the original question but changes almost everything about how it's answered:

| | Original project | This version |
|---|---|---|
| Frequency | Monthly (~37 obs) | Weekly (~230-250 obs) |
| Model | OLS (linear) | Random Forest & Gradient Boosting |
| Instability variable | Binary "war declared" dummy | Caldara-Iacoviello Geopolitical Risk (GPR) index |
| Validation | Standard regression diagnostics | Chronological split, TimeSeriesSplit CV, permutation importance, block-bootstrap CI, tail-conditional R² |

---

## Initial hypothesis

Nassim Taleb divides the world into two separate domains: Mediocristan and Extremistan. While Mediocristan deals with Gaussian type of randomness, Extremistan suffers from a more unstandard random weather and, specially relevant in finance, it suffers from huge tail shocks that would be unconceivable under Gaussian/Mediocristan assumptions even in thousands of years. Our bet is, then: **since we are dealing with modern finances, a linear model should find almost nothing because the relationship between macro conditions and OPEC-linked returns is nonlinear and regime-dependent, not because there's nothing there.** Oil markets are exactly the kind of fat-tailed, shock-driven system where a handful of tail events (a war, a surprise OPEC+ cut, a sanctions regime) should dominate realised returns far more than any smoothly-varying macro beta. If that's true, three things should show up in the results:

1. **Tree ensembles should out-predict a linear model** on the same broad information set, because they can represent threshold effects and interactions an OLS coefficient structurally cannot (e.g. "Brent moves matter more when GPR is already elevated").
2. **A geopolitical-risk index should carry tangible, even if imperfect, explanatory power** —> since we are dealing with an Extremistan environment, we expect this measure to be at least in the top 3 of the 9 variables we derived, which are more classical and some of them (e.g. HML) have been used in this analysis only because of classical financial theory.  
3. **Whatever predictive skill shows up on average will evaporate or invert in the tail** —> e.g. a model that looks fine across a full sample should still fail badly in the worst weeks, because those are precisely the weeks driven by the kind of discrete, hard-to-featurize shock this whole project is about. A good average-case R² should not be mistaken for a good risk model.

The rest of this README is essentially a test of those three claims against what the code actually produced.

---

## Data & methodology

**Ticker panel** — real, independently-listed tickers where they exist; one residual ETF proxy (XOP, "US_proxy") standing in for OPEC producers without a reachable independent listing:

| Company | Ticker | Market |
|---|---|---|
| Aramco | 2222.SR | Tadawul (Saudi Arabia) |
| Petrobras | PBR | NYSE ADR (Brazil) |
| Rosneft | ROSN.ME | MOEX (Russia) |
| MISC Berhad | 3816.KL | Bursa Malaysia (owned by/proxy for Petronas) |
| IPG | IPG.KW | Boursah Kuwait |
| Seplat Energy | SEPL.L | LSE (Nigeria) |
| US_proxy | XOP | US sector ETF, generic OPEC-producer stand-in |
    - These are the only major publicly-traded oil companies that we could find in OPEC+ countries. Regarding XOP, the initial idea was to find an ETF close to how the rest of countries' sectors would behave, but since we did not find any, we thought it would be useful to compare the behavior of this small basket of OPEC's companies and compare it with the Western oil sector, and we chose this S&P-based ETF to model how the US oil sector would behave in the same circumstances as the other tickers.

**Features** (weekly): 
- Market excess return
- HML factor from the Fama-French framework
- Brent crude return
- U.S. crude production % change (EIA) -> hypothesis: it will be negatively-correlated to OPEC's returns, since the US oil sector poses an obvious substitution threat when OPEC decides to restrict oil output to the world. 
- USD index return (related to oil pricing changes, not really operational or geopolitical constraints)
- 10Y-2Y yield curve slope (FRED)
- A shipping-sector ETF proxy (an average between BOAT and SEA, standing in for the paywalled Baltic Dirty Tanker Index)
- The company's own lagged return (momentum)
- `gpr_instability`: the Caldara-Iacoviello GPR index, forward-filled from monthly to weekly. We have used the global index instead of the country-specific ones for each company since both Kuwait and Nigeria were missing in the Caldara-Iacoviello dataset, and furthermore, we observed almost Pareto-efficient results overall when using the simple global index instead of iterating through different countries'. 
    - A **quick note** on the GPR index -> because of the (one could assume) high usefulness and specificity of a geopolitical instability index applied to this domain, we found close to absolutely no free data resource that we could use in this project, either via an API or more conventional data frameworks. This index was the only (free) measure that stood out, with the only (though important) caveat of its being a monthly measure instead of weekly. _If this project were to be used in real life, I could easily change the instability index for a priced, professional measure that would fulfill all of the desired criteria for this model_.

**Models**: Random Forest and Gradient Boosting, both depth/leaf-constrained to resist overfitting on a small sample, evaluated with:
- A strictly **chronological** train/test split (never random: avoids look-ahead bias)
- Expanding-window `TimeSeriesSplit` cross-validation.
- **Permutation importance** (not impurity-based) feature importance, to avoid the well-known bias toward high-cardinality/correlated features that feature importance presents.
- A **block bootstrap** (not i.i.d.) 95% CI on test R², to respect autocorrelation in weekly returns.
- And finally a **tail-conditional R²** on the worst 5% of test weeks, which is the whole point of the exercise, evaluated separately from the headline number.

---

## Results

### Diagnostics run before any model saw the data

- **Multicollinearity**: no explanatory-variable pair exceeded |correlation| > 0.6 — the feature set is reasonably orthogonal, so no variable needed to be dropped or merged going in.
- **Proxy-vs-real correlation** (does XOP/"US_proxy" actually track OPEC-specific economics, or just "being an oil stock"?):

| vs. real ticker | correlation with US_proxy (XOP) |
|---|---|
| Petrobras | +0.591 |
| Rosneft | +0.376 |
| Aramco | +0.338 |
| Seplat Energy | +0.364 |
| MISC Berhad | +0.121 |
| IPG | +0.040 |

Mixed picture: moderate co-movement with Petrobras, essentially none with IPG or MISC Berhad. XOP can be indeed used as a general proxy for anti-OPEC behavior except for the existing correlation that exists between XOP and the real tickers due to, probably, brent price fluctuations.

### Out-of-sample R² by company

| Company | Random Forest | Gradient Boosting | Naive (historical mean) | Beats naive? |
|---|---|---|---|---|
| Aramco | +0.060 | −0.070 | −0.030 | yes (RF only) |
| Petrobras | **+0.176** | +0.117 | −0.008 | yes |
| Rosneft | −0.545 | −3.133 | −0.000 | **no** |
| MISC Berhad | +0.034 | −0.069 | −0.001 | marginal (RF) |
| IPG | +0.016 | −0.024 | −0.004 | marginal (RF) |
| Seplat Energy | −0.019 | −0.159 | −0.082 | marginal (RF, still negative) |
| US_proxy | **+0.516** | +0.433 | −0.006 | yes, strongly |

Random Forest beats Gradient Boosting almost everywhere. From a theoretical lens, with a dataset this small and noisy, GBM's typically-lower-bias/higher-variance profile is a liability rather than an asset.
- A comment on Rosneft is provided in next-to-following section "Why Rosneft failed so badly"

### Is any of this statistically real? (block-bootstrap 95% CI on test R²)

This is the part that keeps the R² table above honest. Once autocorrelation-aware uncertainty is attached:

- **Rosneft**: CI [−1.890, −0.470] (RF) and [−5.533, −1.954] (GBM) — **reliably worse than the naive baseline**, not just unlucky.
- **US_proxy**: CI [+0.095, +0.660] (RF) and [+0.153, +0.602] (GBM) — **reliably better than naive**.
- **Every other company** (Aramco, Petrobras, MISC Berhad, IPG, Seplat): the 95% CI straddles zero in both models. The point estimates look encouraging in places (Petrobras +0.176), but the honest conclusion is that the evidence for genuine predictive skill is statistically indistinguishable from noise for five of the seven names.

### The tail-conditional result (one of the points of the project)

For every company and both model classes, R² computed on just the **worst 5% of test weeks** collapses to deeply negative values (e.g. MISC Berhad's Random Forest tail R² is **−822**, Petrobras's is **−242**, and even US_proxy's is **−58**). Random Forest's tail-window mean absolute error is nonetheless *lower* than the naive baseline's in several cases (e.g. Petrobras: 0.056 vs. naive's 0.074), so this isn't simply the naive model tail being worse across the board, but the R² collapse itself is real and consistent, and with only n=3 tail weeks per company (n=1 for Rosneft). This is also what Taleb would criticize about this model: although it tries to avoid Gaussian frameworks, it still fails to perform properly during the worst shocks (or at least as good as in the normal weeks).

Additionally, even from a non-technical perspective, anyone can notice that the major part of the modeled companies would present a very high R² in 2025, but a completely incorrect model in 2026. This also signals that despite our efforts to account for geopolitical shocks, the biggest oil-related shock in recent history has happened right in these last months coinciding with the start of this year 2026. The model, though, could not catch up with these extreme non-linearities. 

### Why Rosneft failed so badly

Rosneft's usable sample collapses to **29 training weeks and just 8 test weeks**, spanning 2021-08-13 to 2022-05-27 (compared to ~200 training / ~51 test weeks for most other companies). This is mostly due to Western sanctions following the invasion of Ukraine in February 2022, which disrupted MOEX data availability through third-party feeds (including Yahoo Finance, the one we used in this project) shortly after this window starts. This also appears in wildly unstable CV folds (Random Forest R² swinging from −3.623 to +0.092 across four folds) and a test set too short to ever hold real diagnostic weight to begin with.

### Feature importance

Ranked by mean permutation importance across all companies (Random Forest):

```
brent_return                    0.305
HML                             0.016
gpr_instability                 0.016
lagged_own_return                0.007
shipping_proxy_return           0.004
yield_curve_slope               0.000
usd_index_return                -0.001
us_oil_production_pct_change    -0.004
excess_mkt                      -0.015
```

Brent crude return dominates everything else by an order of magnitude, the least surprising and most reassuring finding here. `gpr_instability` shows small but genuinely positive importance for Aramco, MISC Berhad, IPG, and Seplat (roughly 0.01–0.12 depending on model).

**One flagged anomaly**: `gpr_instability` shows *exactly* 0.000000 permutation importance (zero mean, zero std across all 30 shuffles) for Rosneft specifically. Given Rosneft's 8-week test window sits almost entirely inside a single GPR reporting month (GPR is monthly, forward-filled to weekly), the feature is very likely close to **constant** across that test set, so shuffling it changes nothing by construction. 

---

## Main conclusions

1. **Hypothesis #1 - tree ensembles beat linear: partially confirmed.** Random Forest beats a naive baseline for most companies and the improvement is *statistically real* (CI excludes zero) for exactly one non-proxy result out of six real companies tested with a meaningful sample (Rosneft's exclusion is a data-availability failure, not evidence against the model class). Read generously, that's weak-to-moderate support, not a decisive win.
2. **Hypothesis #2 - relative importance of GPR: confirmed.** `gpr_instability` carries small positive importance for four of seven companies unlike most of the other variables except, although it is not a dominant feature anywhere (the only one being more dominant is brent).
3. **Hypothesis #3 - tail performance collapses even when average performance looks fine: confirmed.** This holds for every company and both of the models, since the results for the worst 5% of weeks remain completely incorrect. This means that an investment following this prediction model could survive perfectly fine until there is a big shock. Should this shock be a bit greater than usual, it could take the investor out of the game as this model is fragile to tail events (as opposed to robust or even antifragile).
4. **Mention of US_proxy's strong result.** Its R² (+0.516 RF, statistically significant) is the best in the panel by far, but the proxy diagnostic shows XOP correlates only weakly with several of the real OPEC tickers (as low as 0.040 with IPG). By extension, this pipeline predicts generically the US energy sector reasonably well unlike OPEC's members energy sectors, probably because of the circumstantial differences that the US oil industry and that of the OPEC+ operate in. We still need to mention that even if the US oil sector fits pretty well in the narrative accounted for by our model, the chart "predictions_US_proxy.png" is a clear example that no company escapes the most brutal tail shocks. After the war in Iran the prediction is close to worthless even for the US proxy, as this (not-found-in-the-repo) output shows us:
    Cut off before 2026-01-01    | n= 29 weeks | 2025-06-13 to 2025-12-26 | RF R^2: +0.6920 | GBM R^2: +0.6499
    From 2026-01-01 onward       | n= 22 weeks | 2026-01-02 to 2026-05-29 | RF R^2: +0.3434 | GBM R^2: +0.2211
5. **Random Forest continuously beats Gradient Boosting.** As a theoretical remark, RF uses bagging: a technique consisting on averaging a lot of independently-created decision trees, in a move that reduces variance by definition. This is remarkable in a context with high variance (as ours) as this is something that Gradient Boosting does not take into account while using boosting instead of bagging. Each new tree is fit explicitly to the residual errors left by the ensemble built so far, so with huge errors that compound, the overall performance of this model is a solid basis to dismiss its usage for high-variance sectors like the one in this project in favor of random forests.

---

## Theoretically interesting remarks

- **Average-case R² is close to useless as a risk metric here, and that's the point.** This project's tail-conditional evaluation is a direct, small-scale operationalization of Taleb's central complaint about models built and validated on typical conditions: a model can look internally consistent and even genuinely skillful across the bulk of a sample while failing exactly where the stakes are highest. This is structurally the same critique as the sibling project on credit rating agencies (*Rated Safe, Priced Blind*). A model that is "correct" under its own normal-times assumptions can still be the wrong tool for exactly the moments that matter, without anyone in the pipeline making an identifiable error.
- **Zero permutation importance is ambiguous, and that ambiguity is itself worth knowing.** Rosneft's `gpr_instability` result is a clean example of a feature-importance instrument instead of implying that one variable does not matter, it actually means that it had no variance to exploit in this particular test window. Therefore, reading feature-importance charts requires checking *why* a bar is flat, not just observing that it is.
- **Construct validity matters more than metric size.** US_proxy's headline R² is the best number in the whole table, and it's also the number that needs an asterisk the most once you check what should actually be measuring (a US energy ETF, imperfectly standing in for OPEC dynamics). A large effect size answering the wrong question is not a better result than a small effect size answering the right one.
- **Small-N financial ML results are statistically fragile in a way that's easy to miss without deliberately checking.** Five of seven companies show a CI that straddles zero despite occasionally striking point estimates (Petrobras's +0.176 in particular looks good until its own CI is shown to run from −0.168 to +0.300).
- **A geopolitical rupture can break a model two ways at once: through the mechanism it is meant to detect, and through the data pipeline needed to detect it.** This both refers to the case of Rosneft (and pretty much all the other non-listed OPEC companies that we could not model) could not be properly modeled because of lack of data. Additionally, all the models completely fail to predict 2026 respect to 2025, making the whole point of this project useless due to its failure to address shocks not accounted by traditional financial theory.

---

## Limitations

- **The per-company GPR mapping ended up global, not country-specific, in this run.** `COMPANY_TO_GPR_COL` maps every company to the same aggregate `"GPR"` column rather than each company's own country-level series (`GPRC_SAU`, `GPRC_RUS`, etc.). This was a deliberate simplification made partly because Kuwait and Nigeria have no country-specific GPR series to begin with, partly a choice to keep the instability variable consistent across companies rather than half country-specific / half a global fallback, but most importantly because it was the Pareto-efficient choice between a global metric, a country-specific one, and a OPEC-basket one. We found that to make sense as geopolitical shocks would not affect the global oil market only when it comes to instability in the producing countries but rather instability throughout their clients as well, and in this case the "clients" are potentially every country on Earth. - **GPR is monthly, forward-filled to weekly.** It captures regime-level tension, not week-to-week news flow. This presents instability as something that changes only 1/4 of the times (when the month changes) but stays constant 3/4 of the times, which is not accurate in real life. This was a deliberate choice to avoid paying for better alternative resources (all of the better ones are not free), which would undermine the reproducibility of this project.
- **The n=3 (n=1 for Rosneft) tail evaluation is itself statistically thin.** Directionally consistent and worth taking seriously, but no individual tail-R² number here should be read as a precise estimate, but rather depicting the whole picture where this non-linear model still fails to account for huge non-linear effects.
- **US_proxy (XOP) is a US-sector ETF proxy, not an OPEC company**: its strong result could be read as the Western-sphere oil sector working out fundamentally differently than its Global South's counterpart.

## Future paths

- The main future path, specially if working for a private company without the reproducibility constraint, could be an analysis of different (paid) instability APIs with fundamental differences on their conflict anaylsis to figure out what exactly affects the oil sector rather than overusing a general "instability" variable as if every shock would affect OPEC companies in a statistically significant way.
- Investigate whether another more advanced ML model would do a better work in predicting this small basket of OPEC companies, or even the US_proxy.

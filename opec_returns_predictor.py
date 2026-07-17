# =============================================================================
# Predicting OPEC-Linked Equity Returns with Tree Ensembles
# -----------------------------------------------------------------------------
# Origin: this project extends a university group assignment ("OPEC's Returns")
# that ran a CAPM and a multifactor OLS regression on monthly excess returns of
# seven OPEC-member oil companies. The original found a low R-squared overall,
# with only Brent crude and a binary manually-built "war declared" dummy 
# reaching statistical significance.
#
# What this version changes and why:
#   1. Frequency: monthly -> weekly. ~37 monthly observations is too thin for
#      a tree ensemble to learn stable splits; weekly data gives ~5x more rows
#      over the same calendar window without claiming to be high-frequency.
#   2. Model: Random Forest + Gradient Boosting. We deliberately do NOT
#      assume a linear, additive relationship between macro variables and
#      returns. Oil markets are exactly the kind of regime-dependent, fat-
#      tailed system where linear betas are a weak approximation. This is the
#      Taleb-flavoured premise of the whole project: tail events, not average
#      betas, dominate realised returns. Black-swan-style shocks (a war, an OPEC+ 
#      supply surprise, etc) look like NON-linear interaction effects a tree can
#      capture and a linear model structurally cannot.
#   3. Instability variable: instead of a binary instability indicator, continuous
#      Caldara-Iacoviello Geopolitical Risk (GPR) index, associating each country's
#      GPR value with the oil companies registered in them. This is still an 
#      imperfect signal because of the qualitative nature of geopolitics,
#      but it was the best among the free resources we could find.
#   4. New variables: USD index, yield-curve slope, and a shipping-sector
#      equity proxy are added as additional macro-financial channels with a
#      defensible economic story (see section 3 below for the justification
#      of each).
#
# Limitations:
#   - The Caldara-Iacoviello GPR index is published at MONTHLY resolution, not
#     weekly. Each monthly value is forward-filled across the ~4 weekly Friday
#     dates it covers rather than interpolated (see `fetch_gpr_instability`).
#     It also has the limitation that it does not cover every OPEC country, so 
#     we substitute the Nigerian and Kuwaiti values with the global GPR index 
#     at any moment.
#   - Since many OPEC members do not have an independent, publicly-traded oil
#     company, we have taken a proxy of what these could look like on average
#     with a small twist: using the sector ETF "XOP" by substituting these
#     entries. This ETF is based in the US market so it will eventually
#     become a signal for what a regular, publicly-traded western oil company
#     could return to its investors according to our two machine learning
#     algorithms.
#   - The Baltic Dirty Tanker Index (the ideal shipping-sector indicator) has
#     NO free programmatic data source. We substitute two diversified 
#     shipping-sector ETFs (BOAT & SEA) as a proxy for freight conditions.
#     This is broader than a pure crude-tanker index and weaker as a signal;
#     it is not a clean substitute, just the best free one available.
#
# Author note: every "why" comment in this script documents a methodological
# choice.
#
# SETUP -- run once before executing this script:
#   pip install -r requirements.txt
#
# No API keys are required anywhere in this script. Every data source used
# (yfinance, FRED's fredgraph.csv endpoint, Ken French's data library, and
# the Caldara-Iacoviello GPR index via a direct Excel download from
# matteoiacoviello.com) is free and unauthenticated.
# =============================================================================

import nt
import time
from turtle import mode
from weakref import proxy
import zipfile
import io
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore", category=FutureWarning)

# ── GLOBAL CONFIG ─────────────────────────────────────────────────────────────
# Single source of truth for the sample window and key parameters, so changing
# the backtest period means editing one place, not hunting through the script.
START_DATE = "2016-01-01"
END_DATE   = datetime.today().strftime("%Y-%m-%d")
RANDOM_STATE = 88

# yfinance is used for many equity/index/commodity price series. It is wrapped
# in a small retry helper below because Yahoo's endpoint occasionally rate-
# limits rapid sequential calls.
import yfinance as yf


def _download_with_retry(ticker, start, end, max_attempts=3, pause=5.0):
    """
    Thin wrapper around yf.download with retry/backoff.

    Why this exists: yfinance has no built-in retry logic, and pulling ~15
    tickers back-to-back could occasionally trigger an error. We retry a 
    few times with a short pause to make it work if that happens.
    """
    last_err = None
    for attempt in range(max_attempts):
        try:
            try:
                df = yf.download(ticker, start=start, end=end, progress=False,
                                 auto_adjust=True, multi_level_index=False)
            except TypeError:
                df = yf.download(ticker, start=start, end=end,
                                 progress=False, auto_adjust=True)

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df is None or df.empty:
                print(f"    (attempt {attempt+1}/{max_attempts} for '{ticker}': empty response)")
            if df is not None and len(df) > 0:
                return df
            last_err = ValueError(f"Empty dataframe returned for {ticker}")
        except Exception as e:
            last_err = e
        time.sleep(pause)
    print(f"  [WARNING] Could not retrieve data for '{ticker}' after "
          f"{max_attempts} attempts. Last error: {last_err}")
    return pd.DataFrame()


# =============================================================================
# 1. DEPENDENT VARIABLE: WEEKLY RETURNS OF OPEC-LINKED EQUITIES
# =============================================================================
# Real, independently-listed tickers where they exist, as indicated, since many
# OPEC members run their oil sector with a centralized state-owned company that
# usually are not listed on any exchange.
# 
# We initially tried to represent these OPEC members through a proxy ETF, 
# but as we could not find any that explicitly represented this sector, we
# turned out to the sector ETF XOP.
# XOP returns reflect a basket of mostly US E&P companies, which do not share
# the same cost structure, state-ownership dynamics, or sanctions exposure as
# an OPEC+ state oil company. We are testing (not assuming) whether this
# substitution behaves sanely in section 6.
OPEC_TICKERS = {
    "Aramco": "2222.SR",   # Saudi Arabia - Tadawul exchange
    "Petrobras": "PBR",    # Brazil - NYSE ADR
    "Rosneft": "ROSN.ME",  # Russia - MOEX. NOTE: Western sanctions after the
                              # Ukrainian war since 2022 have intermittently
                              # disrupted MOEX data availability through
                              # third-party feeds, including Yahoo Finance.
                              # This will limit Rosneft's data post-war.
    "MISC Berhad": "3816.KL",  # Malaysia - Bursa Malaysia. Subsidiary company
                                # of Petronas dealing with international oil
                                # shipping and logistics. We use it as a proxy
                                # for Petronas (100% state-owned).
    "IPG": "IPG.KW",        # Kuwait - Boursah Kuwait. IPG deals with refining 
                              # and distribution, not upstream exploration, but 
                              # is publicly listed (unlike the state-owned 
                              # giants). We use it as a proxy for Kuwait's oil 
                              # sector.
    "Seplat Energy": "SEPL.L",  # Nigeria - LSE
    "US_proxy": "XOP",      # As explained in the Readme, this is a sector ETF
                             # proxy for the vast majority of OPEC countries,
                             # which don't have relevant public oil companies.  
                             # This ETF is based on US oil-related companies, 
                             # so it will be useful for comparison between the 
                             # western-US oil sector and the OPEC public 
                             # companies.
}

# Maps each company to its own Caldara-Iacoviello GPR country risk, so
# each company's model sees only its own country's geopolitical risk.
COMPANY_TO_GPR_COL = {
    "Aramco": "GPR",
    "Petrobras": "GPR",
    "Rosneft": "GPR",
    "MISC Berhad": "GPR",
    "IPG": "GPR",
    "Seplat Energy": "GPR",
    "US_proxy": "GPR",
}


def build_dependent_variable(start=START_DATE, end=END_DATE):
    """
    Downloads daily adjusted close prices for every OPEC-linked ticker,
    resamples to weekly (Friday close), and converts to simple weekly returns.

    Returns
    -------
    pd.DataFrame indexed by week-ending date, one return column per company.
    Tickers that fail to download (e.g. a delisted/inaccessible MOEX feed)
    are dropped with a printed warning rather than silently propagating NaNs
    through the rest of the pipeline.
    """
    print("\n[1] Downloading OPEC-linked equity prices...")
    weekly_returns = {}

    for name, ticker in OPEC_TICKERS.items():
        df = _download_with_retry(ticker, start, end)
        if df.empty:
            print(f"  -> Dropping '{name}' ({ticker}): no data retrieved.")
            continue
        price = df["Close"]
        weekly_price = price.resample("W-FRI").last()
        weekly_ret = weekly_price.pct_change()
        weekly_returns[name] = weekly_ret
        print(f"  -> '{name}' ({ticker}): {weekly_ret.notna().sum()} weekly observations.")

    panel = pd.DataFrame(weekly_returns)
    panel.index.name = "week_ending"
    return panel


# =============================================================================
# 2. RISK-FREE RATE AND MARKET RETURN (needed to convert raw returns to excess
#    returns, matching the original project's CAPM-style framing)
# =============================================================================
def fetch_risk_free_and_market(start=START_DATE, end=END_DATE):
    """
    Risk-free rate: 13-week US T-bill (^IRX), already quoted as an annualised
    percentage by Yahoo. We convert it to a weekly rate by dividing by 52,
    since the T-bill is sufficiently low-volatility asset for the RF rate to
    be computed this way.

    Market return: S&P 500 (^GSPC), same role as the original project --
    isolate broad equity-market sentiment from oil-specific effects.
    """
    print("\n[2] Downloading risk-free rate and market return...")

    rf_df = _download_with_retry("^IRX", start, end)
    mkt_df = _download_with_retry("^GSPC", start, end)

    rf_weekly = (rf_df["Close"].resample("W-FRI").last() / 100) / 52
    mkt_price_weekly = mkt_df["Close"].resample("W-FRI").last()
    mkt_ret_weekly = mkt_price_weekly.pct_change()

    out = pd.DataFrame({
        "rf": rf_weekly,
        "mkt_return": mkt_ret_weekly,
    })
    out.index.name = "week_ending"
    return out


# =============================================================================
# 3. EXPLANATORY VARIABLES
# =============================================================================
# Each variable below carries an explicit economic justification, because a
# tree ensemble will happily fit noise if you let it. Feature importance is a 
# description of what the model used, not a certificate that the variable was 
# a good idea.

def fetch_hml_factor(start=START_DATE, end=END_DATE):
    """
    HML (High Minus Low), from Ken French's Data Library, WEEKLY frequency.

    Why: same justification as the original project, OPEC firms behave like
    value stocks (high book-to-market, driven by commodity-price swings and
    regional risk rather than growth expectations), so the value-minus-growth
    spread is a plausible common factor.

    Source mechanics: French's site distributes this as a zipped CSV with a
    few header/footer rows of metadata that are not real data. We download, 
    unzip in-memory, and strip the non-numeric rows.
    """
    print("\n[3a] Downloading Fama-French HML factor (weekly)...")
    url = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
           "F-F_Research_Data_Factors_weekly_CSV.zip")
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            csv_name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
            with z.open(csv_name) as f:
                raw = pd.read_csv(f, skiprows=4)
    except Exception as e:
        print(f"  [WARNING] Failed to download Ken French weekly factors: {e}")
        return pd.Series(dtype=float, name="HML")

    # The file uses an unlabeled first column for the date (YYYYMMDD) and
    # terminates with a block of annual data + copyright text below the
    # daily/weekly rows: we keep only rows whose first column parses as an
    # 8-digit date, which cleanly excludes the trailing junk without having
    # to know in advance how many footer rows there are (file length is not
    # stable release to release).
    raw.columns = ["date_raw"] + list(raw.columns[1:])
    raw = raw[raw["date_raw"].astype(str).str.match(r"^\d{8}$")]
    raw["date"] = pd.to_datetime(raw["date_raw"], format="%Y%m%d")
    raw = raw.set_index("date")
    raw["HML"] = pd.to_numeric(raw["HML"], errors="coerce") / 100  # file is in %
    weekly_hml = raw["HML"].resample("W-FRI").last()
    weekly_hml = weekly_hml.loc[start:end]
    return weekly_hml


def fetch_brent_returns(start=START_DATE, end=END_DATE):
    """
    Brent crude futures (BZ=F), weekly returns.

    Why: Brent is the most widely used global benchmark price for crude oil. 
    This is the most obvious variable in the set.
    """
    print("\n[3b] Downloading Brent crude (BZ=F)...")
    df = _download_with_retry("BZ=F", start, end)
    weekly_price = df["Close"].resample("W-FRI").last()
    return weekly_price.pct_change().rename("brent_return")


def fetch_us_oil_production(start=START_DATE, end=END_DATE):
    """
    Weekly U.S. Field Production of Crude Oil (EIA series WCRFPUS2),
    thousand barrels/day, converted to week-over-week % change.
 
    Why: the US is OPEC's largest non-member competitor. A US production
    surge could offset an OPEC+ supply cut and suppress prices regardless
    of what OPEC itself does. This variable lets the model separate "OPEC
    pricing power" effects from "global oversupply" effects.
 
    We pull directly from EIA's own `LeafHandler.ashx` endpoint, which 
    serves the same underlying series, disregarding EIA's full v2 API 
    in order to avoid requiring changing API keys in the code by an external
    user.
 
    The trade-off: EIA serves this as an HTML table in an unusual WIDE
    format, one row per Year-Month, with up to 5 repeating (End Date,
    Value) column pairs (one pair per week of that month), rather than a
    clean long CSV. We parse and reshape this into a standard long
    date-indexed series below. This keeps the "no API key" property intact.
    """
    print("\n[3c] Downloading US crude oil production (EIA WCRFPUS2)...")
    url = "https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?f=W&n=PET&s=WCRFPUS2"
 
    def _safe_float(x):
        # Defensive against both raw strings like "12,300" (with a thousands
        # separator) AND pre-converted numeric types. Pandas' read_html
        # sometimes auto-converts numeric-looking columns depending on
        # version, so we handle either case rather than assume one.
        if pd.isna(x):
            return None
        if isinstance(x, str):
            return float(x.replace(",", ""))
        return float(x)
 
    try:
        tables = pd.read_html(url)
        candidates = [t for t in tables if t.shape[0] > 50 and t.shape[1] >= 11]
        if not candidates:
            raise ValueError(
                f"No table matching the expected data shape (>50 rows, "
                f">=11 columns) was found among {len(tables)} tables on the "
                f"page. Table shapes found: {[t.shape for t in tables]}"
            )
        raw = candidates[0]
    except Exception as e:
        print(f"  [WARNING] Failed to download/parse US oil production from "
              f"EIA: {e}")
        return pd.Series(dtype=float, name="us_oil_production_pct_change")
 
    records = {}
    for _, row in raw.iterrows():
        year_month = row.iloc[0]
        # Skip header/blank/section-divider rows: only real "YYYY-Mon" rows
        # have a hyphen in this exact position. This is the cleanest filter
        # without needing to know the table's exact row count in advance
        # (EIA's row count grows every week as new data is appended).
        if pd.isna(year_month) or "-" not in str(year_month):
            continue
        try:
            year_str, _month_str = str(year_month).split("-")
            year = int(year_str)
        except ValueError:
            continue
 
        n_pairs = (len(row) - 1) // 2
        for i in range(n_pairs):
            end_date_str = row.iloc[1 + i * 2]
            value_str = row.iloc[2 + i * 2]
            if pd.isna(end_date_str) or pd.isna(value_str):
                continue
            try:
                mm, dd = str(end_date_str).split("/")
                full_date = pd.Timestamp(year=year, month=int(mm), day=int(dd))
                records[full_date] = _safe_float(value_str)
            except (ValueError, TypeError):
                # A handful of malformed rows are expected in a 40+ year HTML
                # table maintained by hand-edited tooling on EIA's end --
                # skip silently rather than fail the whole parse over one
                # bad cell.
                continue
 
    if not records:
        print("  [WARNING] EIA table parsed but yielded zero valid rows -- "
              "the table structure may have changed. Inspect the raw HTML "
              "at the URL above.")
        return pd.Series(dtype=float, name="us_oil_production_pct_change")
 
    series = pd.Series(records).sort_index()
    weekly = series.resample("W-FRI").last()
    weekly = weekly.loc[start:end]
    print(f"  -> Parsed {weekly.notna().sum()} weekly observations from EIA.")
    return weekly.pct_change().rename("us_oil_production_pct_change")


def fetch_usd_index(start=START_DATE, end=END_DATE):
    """
    US Dollar Index (DX-Y.NYB), weekly returns.

    Why: oil is priced in USD globally, so a stronger dollar mechanically 
    makes oil more expensive for non-USD buyers even with no change in the
    underlying crude price, which dampens demand and tends to push USD-priced
    commodities (including oil) down.
    """
    print("\n[3d] Downloading USD index (DX-Y.NYB)...")
    df = _download_with_retry("DX-Y.NYB", start, end)
    if df.empty:
        # DX-Y.NYB is occasionally flaky on Yahoo; UUP (an ETF tracking the
        # dollar index) is a reasonable fallback with near-identical behaviour.
        print("  -> Falling back to UUP (Dollar Index ETF proxy)...")
        df = _download_with_retry("UUP", start, end)
    weekly_price = df["Close"].resample("W-FRI").last()
    return weekly_price.pct_change().rename("usd_index_return")


def fetch_yield_curve_slope(start=START_DATE, end=END_DATE):
    """
    10-Year minus 2-Year Treasury yield spread (FRED series T10Y2Y), in
    percentage points, weekly.

    Why: the yield curve slope is a standard proxy for the market's growth/
    recession expectations. A flattening or inverted curve can easily
    precede demand-driven oil sell-offs. Note that the expectations this
    models are not the same information already contained in the 
    contemporaneous market return.

    We use the LEVEL of the slope, not its weekly change, because the slope
    itself (not its short-term wiggle) is the conventional recession-signal
    convention in the macro-finance literature.
    """
    print("\n[3e] Downloading 10Y-2Y yield curve slope (FRED T10Y2Y)...")
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=T10Y2Y"
    try:
        df = pd.read_csv(url, parse_dates=["observation_date"])
        df = df.set_index("observation_date")
        df.columns = ["yield_curve_slope"]
        df["yield_curve_slope"] = pd.to_numeric(df["yield_curve_slope"], errors="coerce")
    except Exception as e:
        print(f"  [WARNING] Failed to download yield curve slope from FRED: {e}")
        return pd.Series(dtype=float, name="yield_curve_slope")

    weekly = df["yield_curve_slope"].resample("W-FRI").last()
    weekly = weekly.loc[start:end]
    return weekly.ffill().rename("yield_curve_slope")


def fetch_shipping_proxy(start=START_DATE, end=END_DATE):
    """
    Shipping-sector equity proxy: average of BOAT (SonicShares Global
    Shipping ETF) and SEA (US Global Sea to Sky Cargo ETF) weekly returns.

    Why: although this is a weaker variable than the others, it was the
    cheapest no-key data source that we could find (the original idea was 
    the Baltic Dirty Tanker which sold via paid terminals only)

    BOAT and SEA are diversified shipping-company ETFs (container ships, dry
    bulk, tankers, etc. so not crude-tankers exclusively), so they capture
    maritime freight stress rather than crude oil transport trends.
    We use the EQUITY return of shipping companies as an indirect proxy for 
    freight-rate conditions: when tanker day-rates spike, tanker-operator 
    profitability and share prices tend to follow, with a lag and with noise 
    from unrelated segments.

    This is included as the most defensible free substitute available, not
    as an equivalent replacement, so treat its eventual feature importance
    with proportionate skepticism.
    """
    print("\n[3f] Downloading shipping-sector ETF proxies (BOAT, SEA)...")
    boat_df = _download_with_retry("BOAT", start, end)
    sea_df = _download_with_retry("SEA", start, end)

    series_list = []
    if not boat_df.empty:
        series_list.append(boat_df["Close"].resample("W-FRI").last().pct_change().rename("BOAT"))
    else:
        print("  -> BOAT unavailable, proceeding with SEA only if available.")
    if not sea_df.empty:
        series_list.append(sea_df["Close"].resample("W-FRI").last().pct_change().rename("SEA"))
    else:
        print("  -> SEA unavailable, proceeding with BOAT only if available.")

    if not series_list:
        print("  [WARNING] Neither BOAT nor SEA could be retrieved.")
        return pd.Series(dtype=float, name="shipping_proxy_return")

    combined = pd.concat(series_list, axis=1)
    return combined.mean(axis=1, skipna=True).rename("shipping_proxy_return")


def fetch_gpr_instability(start=START_DATE, end=END_DATE):
    """
    Caldara-Iacoviello Geopolitical Risk Index for country resampled to 
    weekly frequency via forward-fill within each month.

    Source: Iacoviello (Fed), freely downloadable Excel, no API key, no
    rate limits, updated monthly around the 1st of each month.
    Cite as: Caldara and Iacoviello (2022), AER 112(4), 1194-1225.
    https://www.matteoiacoviello.com/gpr.htm

    Country-specific GPR columns available in the Excel file
    that overlap with the ticker panel:
        GPRC_SAU  -- Saudi Arabia  (Aramco)
        GPRC_BRA  -- Brazil        (Petrobras)
        GPRC_RUS  -- Russia        (Rosneft)
        GPRC_MYS  -- Malaysia      (MISC Berhad)
        Kuwait (IPG) does not appear in the GPR dataset
        Nigeria (Seplat) does not appear in the GPR dataset
        GPRC_USA  -- United STates (XOR, the proxy)

    NOTE: GPR is monthly, not weekly. Each monthly value is forward-filled
    across the ~4 weekly Friday dates it covers. This is a deliberate,
    documented methodological choice: the GPR captures regime-level
    geopolitical tension, not week-to-week news flow. This is a limitation 
    already discussed in the first part of this file and in the README.MD.
    """
    print("\n[3g] Downloading Caldara-Iacoviello GPR instability index...")
    url = "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"

    GPR_COUNTRY_COLS = list(set(COMPANY_TO_GPR_COL.values()))

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        raw = pd.read_excel(io.BytesIO(resp.content), sheet_name=0)
    except Exception as e:
        print(f"  [WARNING] Failed to download GPR data: {e}")
        return pd.DataFrame(columns=GPR_COUNTRY_COLS, dtype=float)

    date_col = [c for c in raw.columns if "month" in c.lower() or "date" in c.lower()]
    if not date_col:
        print("  [WARNING] Could not identify date column in GPR Excel file. "
              "Column names found:", list(raw.columns[:10]))
        return pd.DataFrame(columns=GPR_COUNTRY_COLS, dtype=float)
    raw["_date"] = pd.to_datetime(raw[date_col[0]], errors="coerce")
    raw = raw.dropna(subset=["_date"]).set_index("_date")

    available = [c for c in GPR_COUNTRY_COLS if c in raw.columns]
    missing   = [c for c in GPR_COUNTRY_COLS if c not in raw.columns]
    if missing:
        # Caldara-Iacoviello does not publish a country-specific series for
        # every country (Kuwait and Nigeria are the 2 gaps in this panel).
        # Rather than zero-fill (which would fabricate a "no risk" signal),
        # we substitute the file's GLOBAL (non-country-specific) GPR index
        # as an explicit, documented, weaker stand-in.
        print(f"  [WARNING] These GPR country columns were not found in the "
              f"downloaded file and will be substituted with the global GPR "
              f"index instead (a national-level series is not published for "
              f"these countries): {missing}. Check column names at "
              f"matteoiacoviello.com/gpr.htm")
        if "GPR" in raw.columns:
            for col in missing:
                raw[col] = raw["GPR"]
            available = available + missing
            missing = []
        else:
            print("  [WARNING] Global 'GPR' fallback column also not found "
                  "-- these companies will have no instability feature.")
    if not available:
        print("  [WARNING] No matching GPR country columns found. "
              "Returning empty DataFrame.")
        return pd.DataFrame(columns=GPR_COUNTRY_COLS, dtype=float)

    for c in available:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")

    monthly_df = raw[available].loc[start:end]

    weekly_idx = pd.date_range(start, end, freq="W-FRI")
    weekly_df = monthly_df.reindex(weekly_idx.union(monthly_df.index))
    weekly_df = weekly_df.ffill().reindex(weekly_idx)

    print(f"  -> GPR: {weekly_df.notna().any(axis=1).sum()} weekly rows across "
          f"{len(available)} country columns: {available}")
    return weekly_df



# =============================================================================
# 4. ASSEMBLE MASTER DATASET
# =============================================================================
def assemble_dataset():
    """
    Pulls every series defined above and joins them on the weekly date index.

    Design choice: we join on the OUTER index of all explanatory variables
    first, THEN attach the dependent variable panel. This way, a gap in one
    explanatory series doesn't silently truncate the others. Every column
    keeps its own NaN pattern until the final modeling step, where we decide
    explicitly how to handle missingness (see `prepare_model_data`).
    """
    print("\n" + "=" * 70)
    print("ASSEMBLING MASTER DATASET")
    print("=" * 70)

    dependent = build_dependent_variable()
    rf_mkt = fetch_risk_free_and_market()
    hml = fetch_hml_factor()
    brent = fetch_brent_returns()
    us_prod = fetch_us_oil_production()
    usd = fetch_usd_index()
    yc_slope = fetch_yield_curve_slope()
    shipping = fetch_shipping_proxy()

    instability = fetch_gpr_instability()

    explanatory = pd.concat(
        [rf_mkt, hml, brent, us_prod, usd, yc_slope, shipping, instability],
        axis=1
    )
    explanatory.index.name = "week_ending"

    full = explanatory.join(dependent, how="left")
    return full, dependent.columns.tolist()


def compute_excess_returns(df, company_cols):
    """
    Converts raw weekly company returns into excess returns (return - rf),
    matching the original project's CAPM-style dependent variable definition.
    """
    excess = df.copy()
    for col in company_cols:
        excess[f"excess_{col}"] = excess[col] - excess["rf"]
    return excess


def compute_excess_market(df):
    """Excess market return, i.e. the classic CAPM right-hand-side term."""
    df["excess_mkt"] = df["mkt_return"] - df["rf"]
    return df


# =============================================================================
# 5. MULTICOLLINEARITY DIAGNOSTIC
# =============================================================================
# Brent returns, the USD index, and the yield-curve slope all move with the 
# same broad risk-on/risk-off macro regime, so a correlation check is run 
# BEFORE modeling, so that any decision to drop or merge a variable is visible
# and justified rather than retrofitted to whatever the model happened to do 
# with it.

def run_multicollinearity_diagnostic(df, feature_cols, save_path="correlation_heatmap.png"):
    """
    Plots a correlation heatmap of the explanatory variables and prints the
    pairs with |correlation| > 0.6 as an explicit flag.

    Note on RF/GBM and multicollinearity: tree ensembles do not break the way
    OLS does when features are correlated (no inflated standard errors, no
    sign flips), but correlated features do split "importance" between
    proxies for the same signal, making the importance ranking harder to
    interpret causally. The diagnostic here is about interpretability, not
    about model validity.
    """
    print("\n" + "=" * 70)
    print("MULTICOLLINEARITY DIAGNOSTIC")
    print("=" * 70)

    corr = df[feature_cols].corr()

    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, square=True, linewidths=0.5)
    plt.title("Correlation Matrix: Explanatory Variables")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  -> Correlation heatmap saved to '{save_path}'")
    plt.close()

    print("\n  Pairs with |correlation| > 0.6 (flagged for interpretability, "
          "not automatically dropped):")
    flagged = False
    for i in range(len(feature_cols)):
        for j in range(i + 1, len(feature_cols)):
            c = corr.iloc[i, j]
            if abs(c) > 0.6:
                print(f"    {feature_cols[i]:25s} <-> {feature_cols[j]:25s}: {c:+.2f}")
                flagged = True
    if not flagged:
        print("    None found -- explanatory set looks reasonably orthogonal.")

    return corr


# =============================================================================
# 6. PROXY-VS-REAL TICKER DIAGNOSTIC
# =============================================================================
# This attempts to treat the proxy-ticker problem as if it weren't
# there at first, then actually check whether the ETF-proxied "companies"
# behave like chaos relative to the real, independently-listed tickers,
# and only remove them if they do.
#
# "Chaos" is operationalised two ways, since neither alone is conclusive:
#   1. Volatility comparison -- do the proxy series show wildly higher/lower
#      weekly return variance than the real tickers, suggesting they're
#      measuring a fundamentally different risk profile?
#   2. Cross-correlation -- do the proxy series correlate AT ALL with the
#      real OPEC tickers? If XOP moves independently of Aramco/Petrobras,
#      that's direct evidence the proxy isn't capturing the same underlying
#      economics, regardless of its own volatility level, assuming that OPEC+
#      member states try to maximize their collective revenue and not just
#      behave individually as in that case, OPEC+ would be a useless cartel.
#
# This function only reports; it does not auto-drop anything. That decision
# is surfaced to you explicitly in the printed output and the final summary.
def run_proxy_diagnostic(df, company_cols):
    print("\n" + "=" * 70)
    print("PROXY-VS-REAL TICKER DIAGNOSTIC")
    print("=" * 70)

    real_cols = [c for c in company_cols if "_proxy" not in c]
    proxy_cols = [c for c in company_cols if "_proxy" in c]

    if not real_cols or not proxy_cols:
        print("  Skipped -- need at least one real and one proxy ticker "
              "with non-empty data to compare.")
        return

    print(f"\n  Real tickers in panel:  {real_cols}")
    print(f"  Proxy tickers in panel: {proxy_cols}")

    print("\n  Weekly return volatility (std. dev.):")
    for c in real_cols + proxy_cols:
        print(f"    {c:20s}: {df[c].std():.4f}")

    print("\n  Correlation of the OPEC (US) proxy with each real ticker:")
    corr_matrix = df[real_cols + proxy_cols].corr()
    for p in proxy_cols:
        for r in real_cols:
            c = corr_matrix.loc[p, r]
            print(f"    {p:20s} vs {r:15s}: corr = {c:+.3f}")

    # Only one "proxy" column (US_proxy, the XOP ETF) remains in this
    # cript, so there is no proxy-vs-proxy correlation to
    # check here. The number that matters is the proxy-vs-real correlation
    # above: if it's close to zero, the proxy isn't standing in for
    # OPEC-specific economics at all, just for "being an oil stock," which
    # is a much weaker claim.
    print("\n  Interpretation guide: a low (<0.3) proxy-vs-real correlation "
          "suggests the ETF proxy is tracking general energy-sector beta "
          "rather than anything specific to OPEC dynamics. ")


# =============================================================================
# 7. MODEL DATA PREPARATION
# =============================================================================
FEATURE_COLS = [
    "excess_mkt",
    "HML",
    "brent_return",
    "us_oil_production_pct_change",
    "usd_index_return",
    "yield_curve_slope",
    "shipping_proxy_return",
]

def prepare_model_data(df, target_col, company_name):
    gpr_col = COMPANY_TO_GPR_COL[company_name]
    lagged_target = df[target_col].shift(1).rename("lagged_own_return")
    df_with_lag = df.join(lagged_target)

    cols_needed = FEATURE_COLS + [gpr_col, "lagged_own_return", target_col]
    model_df = df_with_lag[cols_needed].dropna()
    model_df = model_df.rename(columns={gpr_col: "gpr_instability"})
    X = model_df[FEATURE_COLS + ["gpr_instability", "lagged_own_return"]]
    y = model_df[target_col]
    return X, y, model_df.index

def chronological_train_test_split(X, y, dates, test_size=0.2):
    """
    Splits by TIME, not randomly. This is the most important requirement 
    for any time-series ML pipeline: a random shuffle would let the model 
    train on weeks AFTER the test weeks, which is look-ahead bias. 
    In that case, the model would be implicitly handed a near-answer for 
    many test rows, which is what we are avoiding here. 
    A chronological split closes this off entirely.

    """
    n = len(X)
    split_idx = int(n * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    dates_train, dates_test = dates[:split_idx], dates[split_idx:]
    print(f"  Train: {len(X_train)} weeks ({dates_train[0].date()} to {dates_train[-1].date()})")
    print(f"  Test:  {len(X_test)} weeks ({dates_test[0].date()} to {dates_test[-1].date()})")
    return X_train, X_test, y_train, y_test, dates_train, dates_test


# =============================================================================
# 8. MODELING: RANDOM FOREST AND GRADIENT BOOSTING
# =============================================================================
# Why these two specifically:
#   - Random Forest: an ensemble of de-correlated trees (via bootstrap
#     sampling + random feature subsets at each split). Good baseline for a
#     small-N, noisy financial dataset because it resists overfitting better
#     than a single deep tree, and its variance-reduction mechanism doesn't
#     require careful learning-rate tuning the way boosting does.
#   - Gradient Boosting: builds trees sequentially, each one fitting the
#     residual error of the ensemble so far. Typically lower bias than RF on
#     the same data, at the cost of higher variance / overfitting risk,
#     which is exactly why both are reported side by side here rather than
#     picking one: the original project found a LOW R-squared with OLS, so
#     part of the interesting result either way is whether either tree
#     method actually improves on that, or whether the low explanatory power
#     is a property of the data-generating process itself (i.e. these
#     variables genuinely don't predict OPEC-equity returns well at weekly
#     frequency, full stop) rather than a limitation of linear regression
#     specifically.
#
# Time-series cross-validation: regular k-fold CV is wrong for time-series 
# data as it would fall again in look-ahead bias. sklearn's TimeSeriesSplit
# enforces "train only on the past relative to each fold".
def time_series_cv_score(model, X, y, n_splits=4):
    """
    Returns a list of R-squared scores from expanding-window time-series CV.
    n_splits=4 because the OPEC-returns dataset is far smaller (weekly data 
    over ~3-4 years) and TimeSeriesSplit needs each fold's test set to be
    a meaningful, contiguous future block, not a tiny sliver. Too many
    splits on a short series gives unstable, low-N fold estimates that are
    basically noise rather than signal.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for train_idx, test_idx in tscv.split(X):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[test_idx])
        scores.append(r2_score(y.iloc[test_idx], pred))
    return scores


def bootstrap_r2_ci(y_test, pred, n_bootstrap=2000, block_size=4, random_state=RANDOM_STATE):
    """
    Block bootstrap confidence interval for test-set R^2.

    Why a block bootstrap rather than a standard i.i.d. bootstrap: weekly financial
    returns and their prediction errors are autocorrelated (volatility
    clustering, regime persistence), so resampling individual weeks at
    random would destroy that structure and understate the true uncertainty.
    Resampling contiguous blocks of "block_size" weeks at a time preserves
    short-run dependence while still letting the overall sample vary.

    block_size=4 (~1 month) is a reasonable default for weekly data; there's
    no universally "correct" block size, so report this choice rather than
    treat it as tuned.
    """
    rng = np.random.RandomState(random_state)
    n = len(y_test)
    y_arr = np.asarray(y_test)
    pred_arr = np.asarray(pred)

    n_blocks = int(np.ceil(n / block_size))
    boot_r2s = []

    for _ in range(n_bootstrap):
        starts = rng.randint(0, n - block_size + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_size) for s in starts])[:n]
        boot_r2s.append(r2_score(y_arr[idx], pred_arr[idx]))

    boot_r2s = np.array(boot_r2s)
    lower, upper = np.percentile(boot_r2s, [2.5, 97.5])
    return {
        "point_estimate": r2_score(y_arr, pred_arr),
        "ci_lower": lower,
        "ci_upper": upper,
        "bootstrap_std": boot_r2s.std(),
        "contains_zero": lower <= 0 <= upper,
    }


def train_and_evaluate(company_name, X_train, X_test, y_train, y_test):
    """
    Fits Random Forest + Gradient Boosting, evaluates both against a naive
    "predict the historical mean" baseline, and returns a results dict.

    Why a naive-mean baseline matters here specifically: with weekly equity
    returns, the historical mean is close to zero, so a model needs to clear
    a low bar in absolute MSE terms just by predicting near-zero every week.
    Comparing R-squared against this baseline (rather than just reporting a
    raw MSE number) is the only way to tell whether the model is actually
    learning structure or just exploiting the fact that returns are usually
    small in magnitude.
    """
    print(f"\n{'-' * 70}")
    print(f"Modeling: {company_name}")
    print(f"{'-' * 70}")

    # n_estimators kept modest (200) given the small sample size; more trees
    # mainly help variance reduction, which has diminishing returns well
    # before 200 on a dataset this size, and needlessly inflate run time.
    # max_depth is capped to guard against trees memorising individual weeks
    # in a dataset this small (parallel to the original project's
    # max_leaf_nodes cap on the CART model, same overfitting concern,
    # different mechanism).
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=4,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
    )
    gbm = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
    )

    print("  Running time-series cross-validation (R-squared per fold)...")
    rf_cv_scores = time_series_cv_score(rf, X_train, y_train)
    gbm_cv_scores = time_series_cv_score(gbm, X_train, y_train)
    print(f"    Random Forest CV R^2:     {[f'{s:.3f}' for s in rf_cv_scores]}")
    print(f"    Gradient Boosting CV R^2: {[f'{s:.3f}' for s in gbm_cv_scores]}")

    # Final fit on the full training set (CV above is for model assessment,
    # not for selecting the deployed model's parameters here).
    rf.fit(X_train, y_train)
    gbm.fit(X_train, y_train)

    rf_pred = rf.predict(X_test)
    gbm_pred = gbm.predict(X_test)
    naive_pred = np.repeat(y_train.mean(), len(y_test))

    # Block-bootstrap 95% CI on test R^2, since a single point estimate on
    # ~35-40 weekly rows says nothing about how stable that number is.
    rf_ci = bootstrap_r2_ci(y_test, rf_pred)
    gbm_ci = bootstrap_r2_ci(y_test, gbm_pred)
    print(f"    Random Forest 95% CI on R^2:     [{rf_ci['ci_lower']:+.3f}, {rf_ci['ci_upper']:+.3f}]"
          f"  (point: {rf_ci['point_estimate']:+.3f}, contains zero: {rf_ci['contains_zero']})")
    print(f"    Gradient Boosting 95% CI on R^2: [{gbm_ci['ci_lower']:+.3f}, {gbm_ci['ci_upper']:+.3f}]"
          f"  (point: {gbm_ci['point_estimate']:+.3f}, contains zero: {gbm_ci['contains_zero']})")

    results = {}
    for label, pred in [("Random Forest", rf_pred),
                        ("Gradient Boosting", gbm_pred),
                        ("Naive (historical mean)", naive_pred)]:
        mse = mean_squared_error(y_test, pred)
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        results[label] = {"MSE": mse, "MAE": mae, "R2": r2}
        print(f"    {label:25s} | Test MSE: {mse:.6f} | Test MAE: {mae:.6f} | Test R^2: {r2:+.4f}")

    # Tail-conditional check: does the model's apparent skill hold up in the
    # worst 5% of weeks specifically, or only in the calm, easy-to-predict
    # majority? This is the Taleb-relevant evaluation the headline full-
    # sample R^2 above cannot answer on its own: a model can post a
    # positive average R^2 while doing nothing useful (or actively harmful)
    # exactly in the weeks that matter most for a tail-risk framing.
    n_tail = max(1, int(np.ceil(len(y_test) * 0.05)))
    tail_idx = y_test.sort_values().index[:n_tail]  # 5% worst actual weeks
    y_tail = y_test.loc[tail_idx]

    tail_results = {}
    for label, pred in [("Random Forest", pd.Series(rf_pred, index=y_test.index)),
                        ("Gradient Boosting", pd.Series(gbm_pred, index=y_test.index)),
                        ("Naive (historical mean)", pd.Series(naive_pred, index=y_test.index))]:
        pred_tail = pred.loc[tail_idx]
        tail_r2 = r2_score(y_tail, pred_tail) if n_tail > 1 else float("nan")
        tail_mae = mean_absolute_error(y_tail, pred_tail)
        tail_results[label] = {"tail_R2": tail_r2, "tail_MAE": tail_mae, "n_tail_weeks": n_tail}
        print(f"    {label:25s} | Tail(5%) MAE: {tail_mae:.6f} | Tail(5%) R^2: {tail_r2:+.4f} "
              f"| n={n_tail}")


    return {
        "rf_model": rf,
        "gbm_model": gbm,
        "rf_pred": rf_pred,
        "gbm_pred": gbm_pred,
        "metrics": results,
        "rf_ci": rf_ci, 
        "gbm_ci": gbm_ci,
        "tail_metrics": tail_results,  # 5% worst weeks comparison
    }


# =============================================================================
# 9. FEATURE IMPORTANCE
# =============================================================================
def plot_feature_importance(model, X_test, y_test, company_name, model_label, save_dir="."):
    """
    Uses PERMUTATION importance rather than the model's built-in
    .feature_importances_ attribute.

    Why this matters: sklearn's default impurity-based feature_importances_
    is known to be biased toward high-cardinality/correlated features, which
    can misleadingly give high importance to a variable purely because it
    was available to split on first, not because it carries unique signal.
    Permutation importance instead measures the actual drop in test-set
    performance when a feature's values are shuffled: a more honest
    measure of "does removing this feature's information hurt predictions,"
    which is the right question to ask when several macro variables (Brent,
    USD, yield curve) might be capturing overlapping information.
    """
    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=30, random_state=RANDOM_STATE, scoring="r2"
    )
    importances_df = pd.DataFrame({
        "Feature": X_test.columns,
        "Importance (mean)": result.importances_mean,
        "Importance (std)": result.importances_std,
    }).sort_values("Importance (mean)", ascending=False)

    print(f"\n  Permutation importance -- {company_name} ({model_label}):")
    print(importances_df.to_string(index=False))
    print("  (Negative values are expected and not a bug: on a small test "
          "set, shuffling an uninformative feature can occasionally improve "
          "the score by chance. Treat near-zero and negative values alike "
          "as 'this feature isn't pulling weight,' not as a precise ranking "
          "between them.)")

    plt.figure(figsize=(8, 5))
    plt.barh(importances_df["Feature"], importances_df["Importance (mean)"],
             xerr=importances_df["Importance (std)"], color="#a51c1c")
    plt.xlabel("Permutation Importance (drop in test R^2 when shuffled)")
    plt.title(f"{company_name}: {model_label} Feature Importance")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    fname = f"{save_dir}/feature_importance_{company_name}_{model_label.replace(' ', '_')}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  -> Saved to '{fname}'")

    return importances_df


def plot_predictions_vs_actual(y_test, rf_pred, gbm_pred, company_name, save_dir="."):
    """
    Self-contained comparison plot: actual excess returns vs. both models'
    predictions over the test window. Mirrors the original project's
    practice of embedding the key performance number (here, both models'
    R^2) directly where the reader is already looking, rather than forcing
    them to cross-reference a separate metrics table.
    """
    plt.figure(figsize=(11, 5))
    plt.plot(y_test.index, y_test.values, label="Actual", color="black",
              linewidth=1.8, marker="o", markersize=3)
    plt.plot(y_test.index, rf_pred,
              label=f"Random Forest (R²={r2_score(y_test, rf_pred):.3f})",
              color="#1c6ba5", linewidth=1.3, linestyle="--")
    plt.plot(y_test.index, gbm_pred,
              label=f"Gradient Boosting (R²={r2_score(y_test, gbm_pred):.3f})",
              color="#a51c1c", linewidth=1.3, linestyle="--")
    plt.axhline(0, color="gray", linewidth=0.7)
    plt.title(f"{company_name}: Actual vs. Predicted Weekly Excess Returns (Test Set)")
    plt.ylabel("Weekly excess return")
    plt.legend()
    plt.tight_layout()
    fname = f"{save_dir}/predictions_{company_name}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  -> Saved to '{fname}'")


# =============================================================================
# 10. MAIN EXECUTION
# =============================================================================
def main():
    full_df, company_cols = assemble_dataset()

    full_df = compute_excess_returns(full_df, company_cols)
    full_df = compute_excess_market(full_df)

    # Multicollinearity diagnostic runs on the explanatory set BEFORE any
    # model sees the data, per the user's explicit request.
    run_multicollinearity_diagnostic(full_df, FEATURE_COLS)

    # Proxy-vs-real ticker diagnostic, also per explicit request: report,
    # don't auto-remove.
    run_proxy_diagnostic(full_df, company_cols)

    all_results = {}
    all_importances = {}

    for company in company_cols:
        target_col = f"excess_{company}"
        if target_col not in full_df.columns:
            continue

        X, y, dates = prepare_model_data(full_df, target_col, company)

        if len(X) < 30:
            print(f"\n[SKIPPING '{company}'] Only {len(X)} usable rows after "
                  f"dropping missing data --> too few for a meaningful train/"
                  f"test split. Check the [1] download log above for this "
                  f"company's data availability.")
            continue

        X_train, X_test, y_train, y_test, dates_train, dates_test = \
            chronological_train_test_split(X, y, dates)

        result = train_and_evaluate(company, X_train, X_test, y_train, y_test)
        all_results[company] = result

        rf_importance = plot_feature_importance(
            result["rf_model"], X_test, y_test, company, "Random Forest"
        )
        gbm_importance = plot_feature_importance(
            result["gbm_model"], X_test, y_test, company, "Gradient Boosting"
        )
        all_importances[company] = {"rf": rf_importance, "gbm": gbm_importance}

        plot_predictions_vs_actual(y_test, result["rf_pred"], result["gbm_pred"], company)

    print_summary(all_results, all_importances)
    return full_df, all_results, all_importances


# =============================================================================
# 11. SUMMARY -- INTERPRETATION
# =============================================================================
def print_summary(all_results, all_importances):
    """
    Which variables came out significant, what that might mean, and what to 
    investigate next.
    """
    print("\n" + "=" * 70)
    print("SUMMARY: INTERPRETATION")
    print("=" * 70)

    if not all_results:
        print("  No company had enough data to model. Check the [1] and [3g] "
              "download logs above for missing price or GPR data -- with "
              "GPR's full multi-decade coverage this is more likely a "
              "genuine ticker-level data-availability gap than a systemic "
              "feature-history limit.")
        return

    print("\n  Out-of-sample R^2 by company and model:")
    for company, result in all_results.items():
        rf_r2 = result["metrics"]["Random Forest"]["R2"]
        gbm_r2 = result["metrics"]["Gradient Boosting"]["R2"]
        naive_r2 = result["metrics"]["Naive (historical mean)"]["R2"]
        beat_naive = "yes" if max(rf_r2, gbm_r2) > naive_r2 else "no"
        print(f"    {company:20s} | RF: {rf_r2:+.3f} | GBM: {gbm_r2:+.3f} | "
              f"Naive: {naive_r2:+.3f} | Beat naive baseline: {beat_naive}")

    print("\n  Most consistently important features across companies "
          "(by mean permutation importance, Random Forest):")
    combined_importance = pd.concat(
        [v["rf"].set_index("Feature")["Importance (mean)"].rename(k)
         for k, v in all_importances.items()],
        axis=1
    )
    avg_importance = combined_importance.mean(axis=1).sort_values(ascending=False)
    print(avg_importance.to_string())

    print("""
  Interpretation:

  - If out-of-sample R^2 is low/negative across the board: this is 
    consistent with a Talebian reading that these returns are dominated by 
    unpredictable, fat-tailed shocks (wars, surprise OPEC+ decisions)
    rather than by smoothly varying macro fundamentals. In these cases, no 
    model class (linear or tree-based), should be expected to explain much 
    of the week-to-week variation from public macro data alone.
  - If RF/GBM clearly beat the naive baseline where OLS could not reach
    significance: this would suggest genuine non-linear interaction effects,
    which is exactly the kind of regime-dependent effect a linear model
    cannot represent but a tree can split on directly).
  - Check the proxy-vs-real diagnostic output above before trusting the
    "US_proxy" result -- if XOP showed near-zero correlation with the real
    OPEC tickers, treat that one company's result as "predicting a generic
    energy-sector ETF," not as "predicting OPEC company returns."
          
""")


if __name__ == "__main__":
    full_df, results, importances = main()
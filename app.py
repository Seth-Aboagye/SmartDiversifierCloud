# redeploy test 1
# app.py – SmartDiversifier (updated)

import io
import zipfile
import sqlite3
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from pathlib import Path  # NEW

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# ------------------------------------------------------------------
# DB PATH NOW ANCHORED TO THIS FILE'S FOLDER (WORKS ON CLOUD & LOCAL)
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "smartdiv.db"

TRADING_DAYS = 252

# -----------------------------------------------------------------------------
# 1. OBJECTIVES / METRICS / GUIDE INFO
# -----------------------------------------------------------------------------

# Very simple objective names (kid-friendly)
OBJECTIVE_MAP: Dict[str, List[str]] = {
    "Grow my money a lot (Growth)": ["Sharpe", "Sortino", "CVaR"],
    "Beat the market (Outperformance)": ["Beta", "Sharpe", "Sortino", "RSI"],
    "Avoid big losses (Protection)": ["Max Drawdown", "CVaR", "Volatility"],
    "Keep risk steady (Stable risk)": ["Volatility", "Beta", "Sharpe"],
    "Ride strong trends (Momentum)": ["RSI", "Sharpe", "Sortino", "Beta"],
    "Income focus (Income & yield)": ["CVaR", "Max Drawdown", "Sharpe", "Sortino", "Beta"],
    "Mix of growth & safety (Balanced)": ["Beta", "Volatility", "CVaR"],
    "Invest with values (Sustainable / ESG)": ["Sharpe", "Sortino", "CVaR", "Max Drawdown"],
}

OBJECTIVE_DESC = {
    "Grow my money a lot (Growth)": "Suited for investors who want higher long-term returns and can tolerate ups and downs.",
    "Beat the market (Outperformance)": "Suited for investors trying to outperform a benchmark index.",
    "Avoid big losses (Protection)": "Suited for very risk-aware investors worried about big drawdowns.",
    "Keep risk steady (Stable risk)": "Suited for investors who care about stable volatility levels.",
    "Ride strong trends (Momentum)": "Suited for investors comfortable following price trends.",
    "Income focus (Income & yield)": "Suited for investors who want smoother income and capital preservation.",
    "Mix of growth & safety (Balanced)": "Suited for investors wanting diversification and balanced risk.",
    "Invest with values (Sustainable / ESG)": "Suited for investors who care about ESG and risk-adjusted efficiency.",
}

# Suggested thresholds (just starting points)
SUGGESTED_THRESHOLDS = {
    "Sharpe": 1.0,
    "Sortino": 1.2,
    "Max Drawdown": 0.15,
    "CVaR": 0.10,
    "Volatility": 0.20,
    "Beta": 1.0,
    "RSI": 70,
}

METRIC_DIRECTION = {
    "Sharpe": True,
    "Sortino": True,
    "Max Drawdown": False,
    "CVaR": False,
    "Volatility": False,
    "Beta": "custom",
    "RSI": "custom",
}

# Some reference tickers (not exhaustive)
YF_TICKERS = [
    "SPY", "AGG", "QQQ", "IWM", "EFA", "EMB", "HFRXEH", "BIMAX", "QLENX", "QMNRX",
]
BBG_TICKERS = [
    "SPX Index", "AGG US Equity", "QQQ US Equity", "IWM US Equity",
    "MXEA Index", "EMB US Equity", "HFRXEH Index",
]

# -----------------------------------------------------------------------------
# 2. DATABASE HELPERS
# -----------------------------------------------------------------------------

@st.cache_resource
def get_conn():
    """Return a shared SQLite connection (cached per process)."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db():
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_name TEXT,
            role TEXT,
            investor_name TEXT,
            investor_type TEXT,
            dob_or_incorp TEXT,
            address TEXT,
            city TEXT,
            state TEXT,
            zip_code TEXT,
            country TEXT,
            objective TEXT,
            benchmarks TEXT,
            key_metrics TEXT,
            rec_text TEXT,
            rsi_status TEXT,
            analysis_type TEXT,
            created_at TEXT
        )
        """
    )
    # very small migration layer in case a table existed without new columns
    existing_cols = [
        row[1] for row in c.execute("PRAGMA table_info(analyses)").fetchall()
    ]
    for col, coltype in [
        ("analysis_type", "TEXT"),
        ("rsi_status", "TEXT"),
        ("created_at", "TEXT"),
    ]:
        if col not in existing_cols:
            c.execute(f"ALTER TABLE analyses ADD COLUMN {col} {coltype}")
    conn.commit()


def save_analysis_row(
    user_name: str,
    role: str,
    investor_name: str,
    investor_type: str,
    dob_or_incorp: str,
    address: str,
    city: str,
    state: str,
    zip_code: str,
    country: str,
    objective: str,
    benchmarks: str,
    key_metrics: str,
    rec_text: str,
    analysis_type: str = "BASE",
    rsi_status: Optional[str] = None,
):
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO analyses (
            user_name, role,
            investor_name, investor_type, dob_or_incorp,
            address, city, state, zip_code, country,
            objective, benchmarks, key_metrics,
            rec_text, rsi_status, analysis_type, created_at
        )
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            user_name,
            role,
            investor_name,
            investor_type,
            dob_or_incorp,
            address,
            city,
            state,
            zip_code,
            country,
            objective,
            benchmarks,
            key_metrics,
            rec_text,
            rsi_status,
            analysis_type,
            datetime.utcnow().isoformat(timespec="seconds"),
        ),
    )
    conn.commit()


def load_analyses() -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT * FROM analyses ORDER BY datetime(created_at) DESC", conn
    )
    return df


# -----------------------------------------------------------------------------
# 3. DATA / METRIC UTILITIES
# -----------------------------------------------------------------------------

def read_zip_to_prices(upload: io.BytesIO) -> Tuple[Dict[str, pd.Series], List[str]]:
    """Read ALL CSVs in a ZIP. Accepts multiple common column names."""
    result = {}
    skipped = []
    with zipfile.ZipFile(upload) as z:
        for info in z.infolist():
            if info.filename.lower().endswith(".csv"):
                try:
                    with z.open(info) as f:
                        df = pd.read_csv(f)
                except Exception:
                    skipped.append(f"{info.filename} (read error)")
                    continue

                # normalize columns
                col_map = {c.lower().strip(): c for c in df.columns}
                # date candidates
                date_key = None
                for cand in ["date", "dates", "pricing date"]:
                    if cand in col_map:
                        date_key = col_map[cand]
                        break
                # price candidates (Yahoo + Bloomberg)
                price_key = None
                for cand in [
                    "price",
                    "close",
                    "adj close",
                    "adj_close",
                    "last price",
                    "last_price",
                    "px_last",
                    "px last",
                ]:
                    if cand in col_map:
                        price_key = col_map[cand]
                        break

                if not date_key or not price_key:
                    skipped.append(f"{info.filename} (no Date/Price column)")
                    continue

                try:
                    s = pd.Series(
                        df[price_key].values,
                        index=pd.to_datetime(df[date_key], errors="coerce"),
                        name=info.filename.split("/")[-1].replace(".csv", ""),
                    )
                    s = s[~s.index.isna()]
                    s.index = s.index.tz_localize(None).normalize()
                    s = s[~s.index.duplicated(keep="last")].sort_index()
                except Exception:
                    skipped.append(f"{info.filename} (parse error)")
                    continue

                result[s.name] = s
    return result, skipped


def fetch_yahoo_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    data = yf.download(tickers, start=start, end=end, auto_adjust=False)
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Adj Close"].dropna(how="all")
        close.columns = [c for c in close.columns]
    else:
        close = data.to_frame(name=tickers[0])
    close.index = close.index.tz_localize(None).normalize()
    return close.sort_index()


def align_prices(prices: Dict[str, pd.Series]) -> pd.DataFrame:
    if not prices:
        return pd.DataFrame()
    df = pd.concat(prices.values(), axis=1, join="inner")
    df = df.sort_index().ffill()
    return df


def daily_returns(prices_df: pd.DataFrame) -> pd.DataFrame:
    rets = prices_df.pct_change(fill_method=None)
    return rets.dropna(how="all")


def annualize_ret_std(rets: pd.Series) -> Tuple[float, float]:
    mu_d = rets.mean()
    sd_d = rets.std()
    mu_a = (1 + mu_d) ** TRADING_DAYS - 1
    sd_a = sd_d * np.sqrt(TRADING_DAYS)
    return mu_a, sd_a


def downside_std(rets: pd.Series, rf_daily: float) -> float:
    downside = rets - rf_daily
    downside = downside[downside < 0]
    return downside.std()


def max_drawdown_from_prices(price: pd.Series) -> float:
    running_max = price.cummax()
    drawdown = (running_max - price) / running_max
    return float(drawdown.max())


def empirical_cvar(rets: pd.Series, alpha: float = 0.95) -> float:
    if rets.empty:
        return np.nan
    losses = -rets.dropna()
    var = losses.quantile(alpha)
    tail = losses[losses >= var]
    return float(tail.mean())


def rsi_series(price: pd.Series, window: int = 14) -> pd.Series:
    delta = price.diff()
    gain = delta.clip(lower=0).rolling(window=window).mean()
    loss = -delta.clip(upper=0).rolling(window=window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi


def macd_series(price: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = price.ewm(span=fast, adjust=False).mean()
    ema_slow = price.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return macd, sig, hist


def beta_vs(ret: pd.Series, bench: pd.Series) -> float:
    valid = ret.dropna().index.intersection(bench.dropna().index)
    x = ret.loc[valid] - ret.loc[valid].mean()
    y = bench.loc[valid] - bench.loc[valid].mean()
    denom = (y ** 2).sum()
    return float((x * y).sum() / denom) if denom != 0 else np.nan


def compute_all_metrics(
    prices_df: pd.DataFrame, primary_benchmark: str, rf_pct: float
) -> pd.DataFrame:
    rf_daily = (1 + rf_pct) ** (1 / TRADING_DAYS) - 1
    rets = daily_returns(prices_df)
    bench_ret = rets[primary_benchmark]

    rows = []
    for col in prices_df.columns:
        p = prices_df[col]
        r = rets[col].dropna()
        ann_ret, ann_vol = annualize_ret_std(r)
        sharpe = (r.mean() - rf_daily) / (r.std() + 1e-12) * np.sqrt(TRADING_DAYS)
        dstd = downside_std(r, rf_daily)
        sortino = (r.mean() - rf_daily) / (dstd + 1e-12) * np.sqrt(TRADING_DAYS)
        mdd = max_drawdown_from_prices(p)
        cvar = empirical_cvar(r, 0.95) * np.sqrt(TRADING_DAYS)
        beta_val = beta_vs(r, bench_ret) if col != primary_benchmark else 1.0
        rsi_last = rsi_series(p).iloc[-1]

        rows.append(
            {
                "Fund": col,
                "Ann Return": ann_ret,
                "Volatility": ann_vol,
                "Sharpe": sharpe,
                "Sortino": sortino,
                "Max Drawdown": mdd,
                "CVaR": cvar,
                f"Beta vs {primary_benchmark}": beta_val,
                "RSI(14)": rsi_last,
            }
        )
    df = pd.DataFrame(rows).set_index("Fund")
    return df


def apply_thresholds(df: pd.DataFrame, keys: Dict[str, Tuple[str, float]], benchmark: str):
    mask = pd.Series(True, index=df.index)
    for metric, (op, val) in keys.items():
        col = metric if metric != "Beta" else f"Beta vs {benchmark}"
        if col not in df.columns:
            continue
        if op == ">=":
            mask &= df[col] >= val
        elif op == "<=":
            mask &= df[col] <= val
        elif op == "between":
            lo, hi = val
            if lo is not None:
                mask &= df[col] >= lo
            if hi is not None:
                mask &= df[col] <= hi
    return df[mask]


def plain_recommendation(
    filtered: pd.DataFrame, objective: str, primary_benchmark: str
) -> str:
    if filtered.empty:
        return (
            f"No funds met the thresholds for '{objective}'. "
            "Try relaxing one or two limits or widening the date window."
        )
    if "Sharpe" in filtered.columns:
        top = filtered.sort_values("Sharpe", ascending=False).head(3)
        best = top.index[0]
        why = "highest Sharpe (risk-adjusted return)"
    else:
        top = filtered.sort_values("CVaR", ascending=True).head(3)
        best = top.index[0]
        why = "lowest CVaR (tail risk)"
    note = f"Top candidates: {', '.join(top.index.tolist())}. Best: **{best}** due to {why}. "
    deff = (
        f"For diversification, combine 2–3 shortlisted funds with low correlation and low beta vs {primary_benchmark}."
    )
    return note + deff


# -----------------------------------------------------------------------------
# 4. RSI EXECUTABLE LOGIC
# -----------------------------------------------------------------------------

def rsi_executable_status(price: pd.Series) -> Tuple[str, str]:
    """Return (status, explanation) for the 3-layer RSI/MACD framework."""
    rsi = rsi_series(price)
    rsi_last = rsi.iloc[-1]
    macd, sig, hist = macd_series(price)
    macd_last, sig_last = macd.iloc[-1], sig.iloc[-1]

    # Layer 1: technical alert
    tech_alert = False
    tech_reason = []
    if rsi_last > 80:
        tech_alert = True
        tech_reason.append("RSI is above 80 (overbought).")
    elif rsi_last < 20:
        tech_alert = True
        tech_reason.append("RSI is below 20 (oversold).")
    if np.sign(macd_last) != np.sign(sig_last):
        tech_alert = True
        tech_reason.append("MACD has crossed its signal line (trend change).")

    # Simple confirmation layer based on recent drawdown & volatility
    rets = price.pct_change().dropna()
    mdd = max_drawdown_from_prices(price)
    vol = rets.std() * np.sqrt(TRADING_DAYS)
    confirm_alert = False
    confirm_reason = []
    if mdd > 0.2:
        confirm_alert = True
        confirm_reason.append("Drawdown is larger than 20%.")
    if vol > 0.25:
        confirm_alert = True
        confirm_reason.append("Volatility is higher than 25% annualized.")

    if not tech_alert and not confirm_alert:
        status = "Normal"
        msg = "Signals are calm. No action needed."
    elif tech_alert and not confirm_alert:
        status = "Pre-warning"
        msg = "Technical signals are noisy, but risk measures are still acceptable. Watch closely."
    elif tech_alert and confirm_alert:
        status = "Risk Confirmed"
        msg = "Both technical signals and risk measures look stressed. Consider reducing exposure and hedging."
    else:  # not tech_alert and confirm_alert (rare)
        status = "Review"
        msg = "Risk measures look stressed, but technical signals are quiet. Review the position qualitatively."

    full_reason = " ".join(tech_reason + confirm_reason)
    explanation = f"{msg} {full_reason}"
    return status, explanation


# -----------------------------------------------------------------------------
# 5. STREAMLIT PAGES
# -----------------------------------------------------------------------------

def sidebar_user_block():
    st.sidebar.markdown("### User info (for history only)")
    user_name = st.sidebar.text_input("Your name", key="sb_user_name")
    role = st.sidebar.selectbox(
        "Role", ["Analyst", "Administrator"], key="sb_role_select"
    )
    st.sidebar.markdown("---")
    page = st.sidebar.radio("Navigate", ["Analysis", "RSI Framework", "Guide", "Administrator"], key="sb_nav")
    return user_name, role, page


def page_analysis(user_name: str, role: str):
    st.header("SmartDiversifier – Main Analysis")

    # ----------------------------- DATA SOURCE --------------------------------
    st.subheader("Step 1 – Choose and load data")

    source = st.radio(
        "How do you want to load data?",
        ["Upload ZIP of CSVs", "Fetch from Yahoo Finance"],
        key="src_choice",
    )

    prices_df = None
    skipped = []

    if source == "Upload ZIP of CSVs":
        up = st.file_uploader(
            "Upload ZIP with CSVs (must contain Date and Price/Close columns)", type=["zip"], key="zip_up"
        )
        if up is not None:
            series_dict, skipped = read_zip_to_prices(up)
            prices_df = align_prices(series_dict)
            st.success(
                f"Loaded {prices_df.shape[1]} funds | "
                f"{prices_df.index.min().date()} → {prices_df.index.max().date()} | "
                f"{len(prices_df):,} rows"
            )
            if skipped:
                with st.expander("Files skipped (for reference)"):
                    for s in skipped:
                        st.write("•", s)
            st.session_state["prices_df"] = prices_df
            st.session_state["data_source"] = "zip"
    else:
        c1, c2 = st.columns(2)
        with c1:
            tickers_text = st.text_input(
                "Enter tickers (comma-separated)", value="BIMAX, HFRXEH, QLENX, QMNRX", key="yf_tickers"
            )
        with c2:
            start = st.date_input("Start date", value=pd.to_datetime("2015-01-01"), key="yf_start")
            end = st.date_input("End date", value=pd.to_datetime("today"), key="yf_end")

        if st.button("Fetch from Yahoo Finance", key="btn_fetch_yf"):
            tickers = [t.strip() for t in tickers_text.split(",") if t.strip()]
            if not tickers:
                st.error("Please enter at least one ticker.")
            else:
                df = fetch_yahoo_prices(tickers, start.isoformat(), end.isoformat())
                st.success(
                    f"Fetched {df.shape[1]} tickers from Yahoo | "
                    f"{df.index.min().date()} → {df.index.max().date()} | {len(df):,} rows"
                )
                st.session_state["prices_df"] = df
                st.session_state["data_source"] = "yahoo"

        prices_df = st.session_state.get("prices_df")

    if prices_df is None or prices_df.empty:
        st.info("Load data first to continue with the analysis.")
        return

    st.markdown("#### Price preview")
    st.dataframe(prices_df.tail(10))

    # ------------------------- BENCHMARK CHOICE -------------------------------
    st.subheader("Step 2 – Choose benchmark(s) and risk-free rate")

    benchmarks = st.multiselect(
        "Choose benchmark fund(s)",
        options=list(prices_df.columns),
        default=[prices_df.columns[0]],
        key="bench_multiselect",
    )
    if not benchmarks:
        st.warning("Select at least one benchmark. The first will be used for beta.")
        return
    primary_benchmark = benchmarks[0]

    rf_pct = (
        st.slider(
            "Risk-free rate (annual, %)",
            min_value=0.0,
            max_value=10.0,
            step=0.10,
            value=2.0,
            key="rf_slider",
        )
        / 100.0
    )

    # -------------------------- INVESTOR DETAILS ------------------------------
    st.subheader("Step 3 – Investor details")

    c1, c2, c3 = st.columns(3)
    with c1:
        investor_name = st.text_input("Investor name", key="inv_name")
        investor_type = st.selectbox("Investor type", ["Individual", "Entity"], key="inv_type")
        dob_or_incorp = st.text_input("Date of birth / Incorporation", key="inv_dob")
    with c2:
        address = st.text_input("Address", key="inv_addr")
        city = st.text_input("City", key="inv_city")
        state = st.text_input("State", key="inv_state")
    with c3:
        zip_code = st.text_input("Zip code", key="inv_zip")
        country = st.text_input("Country", value="USA", key="inv_country")

    # --------------------------- OBJECTIVE & KEYS -----------------------------
    st.subheader("Step 4 – Choose investor objective and key metrics")

    objective = st.selectbox("What is the investor trying to achieve?", list(OBJECTIVE_MAP.keys()), key="objective_sel")
    st.write("**Auto-mapped key metrics:**", ", ".join(OBJECTIVE_MAP[objective]))
    st.caption(OBJECTIVE_DESC[objective])

    st.markdown("#### Step 4a – Set thresholds for key metrics")

    mapped = OBJECTIVE_MAP[objective]
    th_cols = st.columns(len(mapped))
    thresholds: Dict[str, Tuple[str, float]] = {}
    for i, m in enumerate(mapped):
        with th_cols[i]:
            default = SUGGESTED_THRESHOLDS.get(m, 0.0)
            direction = METRIC_DIRECTION.get(m, True)
            if m == "RSI":
                mode = st.radio(
                    f"{m} rule", ["≥", "≤", "between 30–70"], index=2, horizontal=True, key=f"rsi_rule_{i}"
                )
                if mode == "≥":
                    val = st.number_input(f"{m} min", value=float(default), step=1.0, key=f"rsi_min_{i}")
                    thresholds[m] = (">=", val)
                elif mode == "≤":
                    val = st.number_input(f"{m} max", value=30.0, step=1.0, key=f"rsi_max_{i}")
                    thresholds[m] = ("<=", val)
                else:
                    thresholds[m] = ("between", (30.0, 70.0))
            elif m == "Beta":
                mode = st.radio(
                    f"{m} rule", ["≤", "≥"], index=0, horizontal=True, key=f"beta_rule_{i}"
                )
                val = st.number_input(
                    f"{m} threshold", value=float(default), step=0.05, key=f"beta_val_{i}"
                )
                thresholds[m] = (mode, val)
            else:
                if direction is True:
                    val = st.number_input(
                        f"{m} ≥", value=float(default), step=0.05, key=f"th_up_{m}_{i}"
                    )
                    thresholds[m] = (">=", val)
                else:
                    val = st.number_input(
                        f"{m} ≤", value=float(default), step=0.01, key=f"th_down_{m}_{i}"
                    )
                    thresholds[m] = ("<=", val)

    # ----------------------------- RUN ANALYSIS -------------------------------
    st.subheader("Step 5 – Run analysis")

    if st.button("▶️ Run Analysis", key="btn_run_main"):
        with st.spinner("Computing metrics and recommendations..."):
            metrics_df = compute_all_metrics(prices_df, primary_benchmark, rf_pct)
            screened = apply_thresholds(metrics_df, thresholds, primary_benchmark)
            rec_text = plain_recommendation(screened, objective, primary_benchmark)

            # Save to session for later (RSI Framework)
            st.session_state["metrics_df"] = metrics_df
            st.session_state["screened_df"] = screened
            st.session_state["primary_benchmark"] = primary_benchmark
            st.session_state["benchmarks"] = benchmarks

            # Save to DB
            if user_name:
                save_analysis_row(
                    user_name=user_name,
                    role=role,
                    investor_name=investor_name,
                    investor_type=investor_type,
                    dob_or_incorp=dob_or_incorp,
                    address=address,
                    city=city,
                    state=state,
                    zip_code=zip_code,
                    country=country,
                    objective=objective,
                    benchmarks=", ".join(benchmarks),
                    key_metrics=", ".join(mapped),
                    rec_text=rec_text,
                    analysis_type="BASE",
                )

        # ------------------ OUTPUT SECTION (RECOMMENDATION FIRST) -------------
        st.success("Analysis complete.")
        st.markdown("### Recommendation (Summary)")
        st.markdown(rec_text)

        st.markdown("---")
        st.markdown("### Full metrics for all funds")
        fmt = {
            "Ann Return": "{:.2%}",
            "Volatility": "{:.2%}",
            "Sharpe": "{:.2f}",
            "Sortino": "{:.2f}",
            "Max Drawdown": "{:.2%}",
            "CVaR": "{:.2%}",
            f"Beta vs {primary_benchmark}": "{:.2f}",
            "RSI(14)": "{:.1f}",
        }
        st.dataframe(metrics_df.style.format(fmt))

        st.markdown("### Funds passing key metric screen")
        st.dataframe(screened.style.format(fmt))

        # ------------------------ CHARTS --------------------------------------
        st.markdown("## Comparison charts")

        # 1) Bar chart by chosen metric – NOW SUPPORTS ALL KEY METRICS
        metric_options = {
            "Sharpe": "Sharpe",
            "Sortino": "Sortino",
            "Annual return": "Ann Return",
            "Volatility": "Volatility",
            "Max drawdown": "Max Drawdown",
            "CVaR": "CVaR",
            f"Beta vs {primary_benchmark}": f"Beta vs {primary_benchmark}",
            "RSI(14)": "RSI(14)",
        }

        metric_label = st.selectbox(
            "Choose metric for bar chart",
            options=list(metric_options.keys()),
            index=0,
            key="chart_metric_sel",
        )
        metric_col = metric_options[metric_label]

        if metric_col not in metrics_df.columns:
            st.warning(f"Metric '{metric_label}' not available in metrics table.")
        else:
            fig_bar = px.bar(
                metrics_df.reset_index(),
                x="Fund",
                y=metric_col,
                title=f"{metric_label} by Fund",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # 2) Risk/return scatter
        fig_scatter = px.scatter(
            metrics_df.reset_index(),
            x="Volatility",
            y="Ann Return",
            text="Fund",
            title="Risk / Return profile",
        )
        fig_scatter.update_traces(textposition="top center")
        st.plotly_chart(fig_scatter, use_container_width=True)

        # 3) Correlation heatmap
        rets = daily_returns(prices_df)
        corr = rets.corr()
        fig_heat = px.imshow(
            corr,
            title="Return correlation heatmap",
            color_continuous_scale="RdBu",
            zmin=-1,
            zmax=1,
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        # 4) Cumulative return lines
        st.markdown("### Cumulative returns (normalized to 100)")
        norm = prices_df / prices_df.iloc[0] * 100
        fig_cum = go.Figure()
        for c in norm.columns:
            fig_cum.add_trace(go.Scatter(x=norm.index, y=norm[c], mode="lines", name=c))
        fig_cum.update_layout(yaxis_title="Index (100 = start)", legend=dict(orientation="h"))
        st.plotly_chart(fig_cum, use_container_width=True)

        # NOTE: Technical view (candlestick + RSI) REMOVED
        # because RSI Framework page provides full technical view.

        # Download metrics
        st.download_button(
            "Download metrics as CSV",
            data=metrics_df.to_csv().encode("utf-8"),
            file_name="smartdiv_metrics.csv",
            mime="text/csv",
        )


def page_rsi_framework(user_name: str, role: str):
    st.header("SmartDiversifier – RSI Executable Framework")

    metrics_df: pd.DataFrame = st.session_state.get("metrics_df")
    prices_df: pd.DataFrame = st.session_state.get("prices_df")
    screened_df: pd.DataFrame = st.session_state.get("screened_df")

    if metrics_df is None or prices_df is None:
        st.warning("Please run the main analysis first, then come back to this page.")
        return

    st.markdown(
        "This page applies a **three-layer RSI / MACD risk check** on a chosen fund, "
        "after the initial objective-based analysis."
    )

    # Funds to choose from: prefer screened funds, fallback to all
    if screened_df is not None and not screened_df.empty:
        default_funds = list(screened_df.index)
        st.info("Showing funds that passed the initial screen. You may still choose others.")
    else:
        default_funds = list(metrics_df.index)

    fund = st.selectbox(
        "Choose fund for RSI Executable analysis",
        options=list(prices_df.columns),
        index=0,
        key="rsi_exec_fund",
    )

    price = prices_df[fund]
    status, explanation = rsi_executable_status(price)

    st.subheader("Status")
    if status == "Normal":
        st.success(f"Status: {status} – {explanation}")
    elif status == "Pre-warning":
        st.warning(f"Status: {status} – {explanation}")
    elif status == "Risk Confirmed":
        st.error(f"Status: {status} – {explanation}")
    else:
        st.info(f"Status: {status} – {explanation}")

    # Charts
    st.markdown("### RSI and MACD charts")

    rsi = rsi_series(price)
    macd, sig, hist = macd_series(price)

    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=rsi.index, y=rsi, mode="lines", name="RSI(14)"))
    fig_rsi.add_hline(y=70, line_dash="dash")
    fig_rsi.add_hline(y=30, line_dash="dash")
    fig_rsi.update_layout(title=f"{fund} – RSI(14)")
    st.plotly_chart(fig_rsi, use_container_width=True)

    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=macd.index, y=macd, mode="lines", name="MACD"))
    fig_macd.add_trace(go.Scatter(x=sig.index, y=sig, mode="lines", name="Signal"))
    fig_macd.add_trace(go.Bar(x=hist.index, y=hist, name="Histogram", opacity=0.3))
    fig_macd.update_layout(title=f"{fund} – MACD")
    st.plotly_chart(fig_macd, use_container_width=True)

    # Save to DB
    if st.button("Save this RSI analysis to history", key="btn_save_rsi"):
        if user_name:
            save_analysis_row(
                user_name=user_name,
                role=role,
                investor_name="(from previous analysis)",
                investor_type="",
                dob_or_incorp="",
                address="",
                city="",
                state="",
                zip_code="",
                country="",
                objective=f"RSI Framework – {fund}",
                benchmarks="",
                key_metrics="RSI, MACD, Drawdown, Volatility",
                rec_text=explanation,
                analysis_type="RSI_EXEC",
                rsi_status=status,
            )
            st.success("RSI analysis saved to Administrator history.")
        else:
            st.warning("Enter your name in the sidebar to save analyses.")


def page_guide():
    st.header("SmartDiversifier – Guide")

    st.markdown("## How to use the Analysis page (front end)")
    st.markdown(
        """
1. **Load data**  
   • Either upload a **ZIP** of CSV files (with Date + Price / Close columns)  
   • Or fetch data directly from **Yahoo Finance** by entering tickers and dates.  

2. **Choose benchmark(s)**  
   • Select one or more benchmark funds.  
   • The **first** benchmark is used to calculate Beta; others are for comparison.  

3. **Enter investor details** (for the history log).  

4. **Select an investor objective** (growth, protection, momentum, etc.).  
   • The app automatically chooses **key metrics** linked to that goal.  

5. **Set metric thresholds**  
   • Use the suggested keys as a starting point.  
   • You can tighten or relax thresholds depending on risk appetite.  

6. **Run the analysis**  
   • The dashboard computes Sharpe, Sortino, Volatility, CVaR, Max Drawdown, Beta, RSI.  
   • It screens funds on the chosen key metrics and compares them to other metrics.  
   • A clear **recommendation** appears at the top.  

7. **Explore charts**  
   • Bar charts, risk/return scatter, correlation heatmap, cumulative returns.  
   • Download full metrics as CSV if needed.
"""
    )

    st.markdown("## How to use the RSI Executable Framework page")
    st.markdown(
        """
1. First run the **main analysis** so prices and metrics are loaded.  
2. Go to **“RSI Framework”**.  
3. Pick a fund (usually from the recommended list).  
4. The app computes:  
   • **RSI** – to detect overbought / oversold levels.  
   • **MACD** – to detect trend changes.  
   • **Drawdown & Volatility** – to confirm whether risk levels are stressed.  

5. The framework then classifies the fund into one of four states:  
   • **Normal** – no alert, normal exposure.  
   • **Pre-warning** – technical alert only, watch more closely.  
   • **Risk Confirmed** – both technical and risk alerts, consider reducing exposure.  
   • **Recovery / Review** – risk easing; rebuild exposure gradually.  

6. You can save each RSI check to the **Administrator history**.
"""
    )

    st.markdown("## Objectives, metrics and investor types")
    data = []
    for obj, metrics in OBJECTIVE_MAP.items():
        data.append(
            {
                "Objective": obj,
                "Aligned metrics": ", ".join(metrics),
                "Suited for": OBJECTIVE_DESC[obj],
            }
        )
    st.dataframe(pd.DataFrame(data))

    st.markdown("## Reference tickers")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Yahoo Finance examples")
        st.write(", ".join(YF_TICKERS))
    with c2:
        st.markdown("### Bloomberg examples")
        st.write(", ".join(BBG_TICKERS))


def page_admin():
    st.header("SmartDiversifier – Administrator history")

    try:
        df = load_analyses()
    except Exception as e:
        st.error(f"Database error: {e}")
        return

    if df.empty:
        st.info("No analyses stored yet.")
        return

    st.markdown("### Full history of analyses")
    st.dataframe(df, use_container_width=True)

    st.markdown("### Download")
    st.download_button(
        "Download history as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="smartdiv_history.csv",
        mime="text/csv",
    )


# -----------------------------------------------------------------------------
# 6. MAIN
# -----------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="SmartDiversifier", layout="wide")
    init_db()

    user_name, role, page = sidebar_user_block()

    if page == "Analysis":
        page_analysis(user_name, role)
    elif page == "RSI Framework":
        page_rsi_framework(user_name, role)
    elif page == "Guide":
        page_guide()
    else:
        page_admin()


if __name__ == "__main__":
    main()

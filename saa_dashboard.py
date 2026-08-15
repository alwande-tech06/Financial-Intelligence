from __future__ import annotations

from pathlib import Path

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
YEARS = list(range(2010, 2020))

# --------------------------------------------------------------------------
# Visual identity: an instrument-panel palette. Cool slate ground, one
# instrument cyan for neutral series, amber and burnt orange reserved
# strictly for deterioration, so colour always carries meaning.
# --------------------------------------------------------------------------
INK = "#0E1621"
SURFACE = "#16202D"
LINE = "#243244"
TEXT = "#E6EDF3"
MUTED = "#8B9AAB"
CYAN = "#4FB3D9"
TEAL = "#2FA37C"
AMBER = "#E9A13B"
ORANGE = "#E4572E"
GOLD = "#C9A227"
VIOLET = "#8C7AE6"

PERIOD_COLOURS = {
    "Pre-scandal": TEAL,
    "Scandal": AMBER,
    "Collapse": ORANGE,
}

st.set_page_config(
    page_title="SAA Financial Intelligence Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600&family=IBM+Plex+Sans:wght@400;500&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    .stApp {{ background: {INK}; color: {TEXT};
              font-family: 'IBM Plex Sans', sans-serif; }}
    section[data-testid="stSidebar"] {{ background: {SURFACE};
              border-right: 1px solid {LINE}; }}
    h1, h2, h3 {{ font-family: 'Barlow Condensed', sans-serif;
              letter-spacing: .045em; text-transform: uppercase;
              color: {TEXT}; font-weight: 600; }}
    h1 {{ font-size: 2.05rem; margin-bottom: .1rem; }}
    h3 {{ font-size: 1.15rem; color: {MUTED}; margin-top: .4rem; }}

    .masthead {{ border-bottom: 1px solid {LINE}; padding-bottom: .7rem;
              margin-bottom: 1rem; }}
    .masthead .sub {{ font-family: 'IBM Plex Mono', monospace;
              font-size: .74rem; color: {MUTED}; letter-spacing: .1em; }}

    .card {{ background: {SURFACE}; border: 1px solid {LINE};
             border-left: 3px solid var(--tone, {CYAN});
             border-radius: 3px; padding: .7rem .85rem; height: 100%; }}
    .card .label {{ font-family: 'Barlow Condensed', sans-serif;
             text-transform: uppercase; letter-spacing: .09em;
             font-size: .78rem; color: {MUTED}; }}
    .card .value {{ font-family: 'IBM Plex Mono', monospace;
             font-size: 1.42rem; font-weight: 600; color: {TEXT};
             line-height: 1.5; }}
    .card .delta {{ font-family: 'IBM Plex Mono', monospace;
             font-size: .74rem; color: {MUTED}; }}

    .flagline {{ font-family: 'IBM Plex Mono', monospace; font-size: .82rem;
             padding: .18rem 0; border-bottom: 1px dotted {LINE}; }}
    .stDataFrame {{ border: 1px solid {LINE}; }}
    div[data-testid="stMetricValue"] {{ font-family: 'IBM Plex Mono', monospace; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Data access
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tidy = pd.read_csv(DATA_DIR / "saa_financials_tidy.csv")
    wide = pd.read_csv(DATA_DIR / "saa_metrics_wide.csv")
    restate_path = DATA_DIR / "saa_restatements.csv"
    restated = pd.read_csv(restate_path) if restate_path.exists() else pd.DataFrame()
    return tidy, wide, restated


def blank_years(frame: pd.DataFrame) -> pd.DataFrame:
    """Insert the missing FY2013/FY2014 so every series breaks honestly."""
    return frame.set_index("year").reindex(YEARS).reset_index()


@st.cache_data(show_spinner=False)
def load_ml() -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load ML outputs produced by saa_ml.py, or return empty structures."""
    def _csv(name):
        p = DATA_DIR / name
        return pd.read_csv(p) if p.exists() else pd.DataFrame()

    meta_path = DATA_DIR / "ml_metrics.json"
    if not meta_path.exists():
        return {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    meta = json.loads(meta_path.read_text())
    return meta, _csv("ml_saa_scores.csv"), _csv("ml_feature_importance.csv"), _csv("ml_roc.csv")


@st.cache_data(show_spinner=False)
def compute_indicators(wide: pd.DataFrame) -> pd.DataFrame:
    """Derive the ratio set used across Question 4.1."""
    d = wide.copy()

    def col(name):
        return d[name] if name in d.columns else pd.Series(np.nan, index=d.index)

    income = col("total_income")
    d["working_capital"] = col("current_assets") - col("current_liabilities")
    d["current_ratio"] = col("current_assets") / col("current_liabilities")
    d["quick_ratio"] = (col("current_assets") - col("inventories")) / col("current_liabilities")
    d["cash_ratio"] = col("cash") / col("current_liabilities")
    d["ebitda_margin"] = col("ebitda") / income * 100
    d["operating_margin"] = col("operating_loss") / income * 100
    d["net_margin"] = col("loss_for_year") / income * 100
    d["cost_to_income"] = col("operating_costs") / income * 100
    d["gearing"] = col("total_liabilities") / col("total_assets") * 100
    d["equity_ratio"] = col("total_equity") / col("total_assets") * 100
    d["interest_cover"] = col("ebitda") / col("finance_costs")
    d["debt"] = col("long_term_loans") + col("current_portion_loans") + col("bank_overdraft").fillna(0)
    d["asset_turnover"] = income / col("total_assets")
    d["days_sales_outstanding"] = col("receivables") / income * 365
    d["accruals_ratio"] = (col("loss_for_year") - col("net_operating_cash")) / col("total_assets") * 100
    d["cash_conversion"] = col("net_operating_cash") / income * 100
    d["capex_to_depreciation"] = col("capex") / col("depreciation")
    d["fuel_share"] = col("fuel_costs") / col("operating_costs") * 100
    d["staff_share"] = col("employee_costs") / col("operating_costs") * 100

    # Altman Z''-score, emerging-market variant for non-manufacturers.
    x1 = d["working_capital"] / col("total_assets")
    x2 = col("accumulated_loss") / col("total_assets")
    x3 = col("operating_loss") / col("total_assets")
    x4 = col("total_equity") / col("total_liabilities")
    d["altman_z"] = 3.25 + 6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4
    d["z_x1_liquidity"], d["z_x2_retained"] = x1, x2
    d["z_x3_earnings"], d["z_x4_solvency"] = x3, x4
    return d


def assign_periods(frame: pd.DataFrame, pre_end: int, scandal_end: int) -> pd.DataFrame:
    def label(year):
        if year <= pre_end:
            return "Pre-scandal"
        if year <= scandal_end:
            return "Scandal"
        return "Collapse"
    out = frame.copy()
    out["period"] = out["year"].apply(label)
    return out


# --------------------------------------------------------------------------
# Formatting and layout helpers
# --------------------------------------------------------------------------
def rm(value, decimals: int = 0) -> str:
    """Format R million the way the annual report does: R26 992m."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    sign = "-" if value < 0 else ""
    text = f"{abs(value):,.{decimals}f}".replace(",", " ")
    return f"{sign}R{text}m"


def num(value, decimals: int = 2, suffix: str = "") -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return f"{value:,.{decimals}f}{suffix}".replace(",", " ")


def kpi(container, label: str, value: str, delta: str = "", tone: str = CYAN) -> None:
    container.markdown(
        f"""<div class="card" style="--tone:{tone}">
              <div class="label">{label}</div>
              <div class="value">{value}</div>
              <div class="delta">{delta}</div>
            </div>""",
        unsafe_allow_html=True,
    )


def show(fig, height: int = 380) -> None:
    """Apply the house style, then render across the container width."""
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=SURFACE,
        font=dict(family="IBM Plex Sans, sans-serif", color=TEXT, size=12),
        margin=dict(l=8, r=8, t=100, b=8),
        title=dict(font=dict(family="Barlow Condensed, sans-serif", size=17,
                             color=TEXT),
                   y=0.97, yanchor="top", x=0, xanchor="left"),
        legend=dict(orientation="h", yanchor="bottom", y=1.06, x=0,
                    font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(font=dict(family="IBM Plex Mono, monospace", size=11)),
    )
    fig.update_xaxes(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE,
                     tickfont=dict(family="IBM Plex Mono, monospace", size=11))
    fig.update_yaxes(gridcolor=LINE, zerolinecolor=MUTED, linecolor=LINE,
                     tickfont=dict(family="IBM Plex Mono, monospace", size=11))
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:  # Streamlit < 1.49
        st.plotly_chart(fig, use_container_width=True)


def table(frame, height: int | None = None) -> None:
    """Render a dataframe across the container width, version-agnostically."""
    kwargs = {"height": height} if height else {}
    try:
        st.dataframe(frame, width="stretch", **kwargs)
    except TypeError:  # Streamlit < 1.49
        st.dataframe(frame, use_container_width=True, **kwargs)


def period_bands(fig, periods: pd.DataFrame) -> None:
    """Shade the scandal timeline behind a time series."""
    for name, group in periods.groupby("period"):
        fig.add_vrect(
            x0=group["year"].min() - 0.5, x1=group["year"].max() + 0.5,
            fillcolor=PERIOD_COLOURS.get(name, MUTED), opacity=0.07,
            layer="below", line_width=0,
            annotation_text=name.upper(),
            annotation_position="top left",
            annotation_font=dict(family="Barlow Condensed, sans-serif",
                                 size=11, color=MUTED),
        )


def delta_text(series: pd.Series, year: int, formatter=rm) -> tuple[str, str]:
    """Change against the previous year that actually exists in the data."""
    available = series.dropna()
    prior = [y for y in available.index if y < year]
    if year not in available.index or not prior:
        return "", CYAN
    last = max(prior)
    change = available[year] - available[last]
    arrow = "▲" if change > 0 else "▼" if change < 0 else "▶"
    tone = CYAN if abs(change) < 1e-9 else (TEAL if change > 0 else ORANGE)
    return f"{arrow} {formatter(abs(change))} vs FY{last}", tone


# --------------------------------------------------------------------------
# Red-flag rule engine (Question 4.1)
# --------------------------------------------------------------------------
FLAG_RULES = [
    ("Loss-making year", lambda r: r.get("loss_for_year", np.nan) < 0),
    ("Negative EBITDA", lambda r: r.get("ebitda", np.nan) < 0),
    ("Negative equity (technically insolvent)", lambda r: r.get("total_equity", np.nan) < 0),
    ("Current ratio below 1.0", lambda r: r.get("current_ratio", np.nan) < 1),
    ("Operating cash outflow", lambda r: r.get("net_operating_cash", np.nan) < 0),
    ("Interest cover below 1.0", lambda r: r.get("interest_cover", np.nan) < 1),
    ("Altman Z'' in distress zone", lambda r: r.get("altman_z", np.nan) < 4.15),
    ("Accruals ratio beyond ±10%", lambda r: abs(r.get("accruals_ratio", np.nan)) > 10),
    ("Liabilities exceed 100% of assets", lambda r: r.get("gearing", np.nan) > 100),
    ("Shareholder bailout received", lambda r: r.get("shareholder_bailout", 0) > 0),
]


def flag_matrix(ind: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for _, record in ind.iterrows():
        rows[int(record["year"])] = {
            name: bool(rule(record)) if pd.notna(rule(record)) else False
            for name, rule in FLAG_RULES
        }
    return pd.DataFrame(rows).T.sort_index()


# --------------------------------------------------------------------------
# Sidebar (the slicer pane)
# --------------------------------------------------------------------------
try:
    tidy, wide, restated = load_data()
except FileNotFoundError:
    st.error(
        "No parsed dataset found. Run the ETL first:\n\n"
        "`python saa_etl.py --source SAA_Financial_statement.xlsx --outdir data`"
    )
    st.stop()

st.sidebar.markdown("### Slicers")
entity = st.sidebar.radio(
    "Reporting entity", ["Group", "Company"], horizontal=True,
    help="Group consolidates Mango, SAA Technical, Air Chefs and SAA Cargo; "
         "Company is SAA SOC Ltd standalone.",
)
year_range = st.sidebar.select_slider(
    "Financial years", options=YEARS, value=(YEARS[0], YEARS[-1]),
)
st.sidebar.markdown("### Scandal timeline")
pre_end = st.sidebar.slider("Pre-scandal period ends", 2011, 2016, 2012)
scandal_end = st.sidebar.slider("Scandal period ends", pre_end + 1, 2019, 2018)

indicators = compute_indicators(wide)
indicators = assign_periods(indicators, pre_end, scandal_end)
ent = indicators[indicators["entity"] == entity].copy()
ent = ent[(ent["year"] >= year_range[0]) & (ent["year"] <= year_range[1])]
ent = ent.sort_values("year").reset_index(drop=True)
series = ent.set_index("year")

focus_years = [int(y) for y in ent["year"]]
focus = st.sidebar.selectbox(
    "Focus year (waterfalls and breakdowns)", focus_years,
    index=len(focus_years) - 1,
)
st.sidebar.markdown(
    f"<div style='font-family:IBM Plex Mono,monospace;font-size:.72rem;"
    f"color:{MUTED};padding-top:.6rem;border-top:1px solid {LINE}'>"
    f"Source: SAA annual financial statements, R million.<br>"
    f"{len(tidy):,} parsed line-item values.<br>"
    f"FY2013 and FY2014 not published in the source workbook.</div>",
    unsafe_allow_html=True,
)

st.markdown(
    """<div class="masthead">
         <h1>South African Airways Financial Intelligence Dashboard</h1>
         <div class="sub">FY2010–FY2019 &nbsp;·&nbsp; FORENSIC ANALYTICS &nbsp;·&nbsp; ALL FIGURES R MILLION</div>
       </div>""",
    unsafe_allow_html=True,
)

tab_overview, tab_profit, tab_position, tab_cash, tab_distress, tab_ml, tab_data = st.tabs(
    ["Overview", "Profitability", "Financial position", "Cash & funding",
     "Distress & red flags", "Predictive analytics", "Data"]
)

# --------------------------------------------------------------------------
# TAB 1 — Overview scorecard
# --------------------------------------------------------------------------
with tab_overview:
    cards = st.columns(6)
    row = series.loc[focus] if focus in series.index else pd.Series(dtype=float)

    spec = [
        ("Total income", "total_income", rm, "higher_better"),
        ("EBITDA", "ebitda", rm, "higher_better"),
        ("Loss for the year", "loss_for_year", rm, "higher_better"),
        ("Cash on hand", "cash", rm, "higher_better"),
        ("Total equity", "total_equity", rm, "higher_better"),
        ("Altman Z''-score", "altman_z", lambda v: num(v, 2), "higher_better"),
    ]
    for container, (label, metric, formatter, _) in zip(cards, spec):
        value = row.get(metric, np.nan)
        change, tone = delta_text(series[metric], focus, formatter) \
            if metric in series.columns else ("", CYAN)
        if metric in {"total_equity", "loss_for_year", "ebitda"} and value < 0:
            tone = ORANGE
        kpi(container, f"{label} · FY{focus}", formatter(value), change, tone)

    st.markdown("")
    left, right = st.columns([3, 2])

    with left:
        plot = blank_years(ent[["year", "total_income", "operating_costs",
                                "loss_for_year"]])
        fig = go.Figure()
        fig.add_bar(x=plot["year"], y=plot["total_income"], name="Total income",
                    marker_color=CYAN, opacity=.85)
        fig.add_bar(x=plot["year"], y=plot["operating_costs"],
                    name="Operating costs", marker_color=MUTED, opacity=.55)
        fig.add_scatter(x=plot["year"], y=plot["loss_for_year"],
                        name="Loss for the year", yaxis="y2", mode="lines+markers",
                        line=dict(color=ORANGE, width=2.5), marker=dict(size=7),
                        connectgaps=False)
        fig.update_layout(
            title="Income, cost base and bottom line",
            barmode="group",
            yaxis=dict(title="R million"),
            yaxis2=dict(title="Loss (R million)", overlaying="y", side="right",
                        showgrid=False, zeroline=True, zerolinecolor=MUTED),
        )
        period_bands(fig, ent[["year", "period"]])
        show(fig, 420)

    with right:
        row_focus = series.loc[focus] if focus in series.index else pd.Series(dtype=float)
        # Plotly ignores y on "total" steps when positioning the bar, so the
        # published subtotal is passed purely so the data label reads correctly.
        steps = [
            ("Income", row_focus.get("total_income", np.nan), "relative"),
            ("Op. costs", -abs(row_focus.get("operating_costs", np.nan)), "relative"),
            ("EBITDA", row_focus.get("ebitda", np.nan), "total"),
            ("D&A", -abs(row_focus.get("depreciation", np.nan)), "relative"),
            ("Impairment", -abs(row_focus.get("impairments", 0) or 0), "relative"),
            ("Net finance", row_focus.get("interest_income", 0)
             - abs(row_focus.get("finance_costs", 0) or 0), "relative"),
            ("Tax", row_focus.get("taxation", 0), "relative"),
            ("Net loss", row_focus.get("loss_for_year", np.nan), "total"),
        ]
        fig = go.Figure(go.Waterfall(
            orientation="v",
            measure=[s[2] for s in steps],
            x=[s[0] for s in steps],
            y=[s[1] for s in steps],
            connector=dict(line=dict(color=LINE)),
            increasing=dict(marker=dict(color=TEAL)),
            decreasing=dict(marker=dict(color=ORANGE)),
            totals=dict(marker=dict(color=CYAN)),
            textposition="outside",
            texttemplate="%{y:,.0f}",
            textfont=dict(family="IBM Plex Mono, monospace", size=10),
        ))
        fig.update_layout(title=f"FY{focus}: how income became loss",
                          yaxis=dict(title="R million"))
        show(fig, 420)

    period_view = (ent.groupby("period")
                   .agg(years=("year", "count"),
                        avg_income=("total_income", "mean"),
                        avg_ebitda=("ebitda", "mean"),
                        avg_loss=("loss_for_year", "mean"),
                        avg_equity=("total_equity", "mean"),
                        avg_z=("altman_z", "mean"))
                   .reindex([p for p in PERIOD_COLOURS if p in set(ent["period"])]))
    st.markdown("### Period comparison")
    table(period_view.style.format({
        "avg_income": "{:,.0f}", "avg_ebitda": "{:,.0f}",
        "avg_loss": "{:,.0f}", "avg_equity": "{:,.0f}", "avg_z": "{:,.2f}",
    }))

    with st.expander("Interpretation — what this view shows"):
        first_loss = ent[ent["loss_for_year"] < 0]["year"].min()
        has_loss = pd.notna(first_loss) and ent["loss_for_year"].notna().any()
        worst = ent.loc[ent["loss_for_year"].idxmin()] if has_loss else None
        if not has_loss:
            st.markdown("- No loss years in the selected range.")
        else:
            st.markdown(
                f"- The cost base overtook total income from FY{int(first_loss)} onward and never "
                f"recovered: {entity} reported a loss in every subsequent published year.\n"
                f"- The deepest loss in the series is FY{int(worst['year'])} at "
                f"{rm(worst['loss_for_year'])}, driven by an operating result of "
                f"{rm(worst['operating_loss'])} before finance costs of "
                f"{rm(worst['finance_costs'])}.\n"
                f"- The waterfall isolates where value is destroyed: the loss is created above "
                f"the EBITDA line, in the operating cost base, not by financing or tax."
            )

# --------------------------------------------------------------------------
# TAB 2 — Profitability and cost structure
# --------------------------------------------------------------------------
with tab_profit:
    left, right = st.columns(2)

    with left:
        plot = blank_years(ent[["year", "ebitda_margin", "operating_margin",
                                "net_margin"]])
        fig = go.Figure()
        for name, column, colour in [
            ("EBITDA margin", "ebitda_margin", CYAN),
            ("Operating margin", "operating_margin", AMBER),
            ("Net margin", "net_margin", ORANGE),
        ]:
            fig.add_scatter(x=plot["year"], y=plot[column], name=name,
                            mode="lines+markers", connectgaps=False,
                            line=dict(color=colour, width=2.4))
        fig.add_hline(y=0, line_color=MUTED, line_width=1)
        fig.update_layout(title="Margin deterioration (% of total income)",
                          yaxis=dict(title="%", ticksuffix="%"))
        period_bands(fig, ent[["year", "period"]])
        show(fig)

    with right:
        cost_cols = ["fuel_costs", "employee_costs", "maintenance_costs",
                     "lease_costs", "navigation_costs", "commission_costs",
                     "catering_costs", "other_operating_costs"]
        labels = ["Fuel & energy", "Employee benefits", "Maintenance",
                  "Aircraft leases", "Navigation & landing", "Commissions",
                  "Catering", "Other operating"]
        colours = [ORANGE, CYAN, VIOLET, AMBER, TEAL, GOLD, MUTED, LINE]
        available = [(c, l, k) for c, l, k in zip(cost_cols, labels, colours)
                     if c in ent.columns]
        plot = blank_years(ent[["year"] + [c for c, _, _ in available]])
        total = plot[[c for c, _, _ in available]].sum(axis=1)
        fig = go.Figure()
        for column, label, colour in available:
            fig.add_bar(x=plot["year"], y=plot[column] / total * 100, name=label,
                        marker_color=colour)
        fig.update_layout(title="Cost structure (% of operating cost base)",
                          barmode="stack", yaxis=dict(title="%", ticksuffix="%"))
        show(fig)

    left, right = st.columns(2)
    with left:
        base_year = ent["year"].min()
        idx = blank_years(ent[["year", "total_income", "operating_costs"]])
        base = idx[idx["year"] == base_year]
        fig = go.Figure()
        for column, name, colour in [("total_income", "Total income", CYAN),
                                     ("operating_costs", "Operating costs", ORANGE)]:
            denominator = base[column].values[0] if len(base) else np.nan
            fig.add_scatter(x=idx["year"], y=idx[column] / denominator * 100,
                            name=name, mode="lines+markers", connectgaps=False,
                            line=dict(color=colour, width=2.4))
        fig.add_hline(y=100, line_color=MUTED, line_dash="dot", line_width=1)
        fig.update_layout(title=f"Income vs cost growth (FY{base_year} = 100)",
                          yaxis=dict(title="Index"))
        show(fig)

    with right:
        row_focus = series.loc[focus] if focus in series.index else pd.Series(dtype=float)
        breakdown = pd.DataFrame({
            "item": [l for c, l, _ in available],
            "value": [row_focus.get(c, np.nan) for c, _, _ in available],
        }).dropna().sort_values("value")
        fig = go.Figure(go.Bar(
            x=breakdown["value"], y=breakdown["item"], orientation="h",
            marker_color=CYAN, text=breakdown["value"].map(lambda v: f"{v:,.0f}"),
            textposition="outside",
            textfont=dict(family="IBM Plex Mono, monospace", size=10),
        ))
        fig.update_layout(title=f"FY{focus} cost drivers (R million)")
        show(fig)

    with st.expander("Interpretation — what this view shows"):
        fuel = series["fuel_share"].dropna()
        staff = series["staff_share"].dropna()
        cti = series["cost_to_income"].dropna()
        if cti.empty or fuel.empty:
            st.markdown("- Insufficient data in the selected range.")
        else:
          st.markdown(
            f"- Cost-to-income moved from {num(cti.iloc[0], 1)}% in FY{int(cti.index[0])} "
            f"to {num(cti.iloc[-1], 1)}% in FY{int(cti.index[-1])}: every rand of income "
            f"was consumed by more than a rand of cost.\n"
            f"- Fuel and employee benefits together account for roughly "
            f"{num(fuel.iloc[-1] + staff.iloc[-1], 0)}% of the cost base in "
            f"FY{int(fuel.index[-1])}, which limits how much of the deficit could be "
            f"closed by discretionary cost-cutting alone.\n"
            f"- The index chart separates a revenue problem from a cost problem: "
            f"income is broadly flat while the cost line climbs, so the deficit is "
            f"structural rather than cyclical."
          )

# --------------------------------------------------------------------------
# TAB 3 — Financial position
# --------------------------------------------------------------------------
with tab_position:
    cards = st.columns(5)
    row_focus = series.loc[focus] if focus in series.index else pd.Series(dtype=float)
    kpi(cards[0], f"Total assets · FY{focus}", rm(row_focus.get("total_assets")),
        *delta_text(series["total_assets"], focus))
    kpi(cards[1], f"Total liabilities · FY{focus}", rm(row_focus.get("total_liabilities")),
        *delta_text(series["total_liabilities"], focus))
    kpi(cards[2], f"Accumulated loss · FY{focus}", rm(row_focus.get("accumulated_loss")),
        "", ORANGE)
    kpi(cards[3], f"Current ratio · FY{focus}", num(row_focus.get("current_ratio"), 2),
        "solvency benchmark 1.00", ORANGE if row_focus.get("current_ratio", 9) < 1 else TEAL)
    kpi(cards[4], f"Gearing · FY{focus}", num(row_focus.get("gearing"), 1, "%"),
        "liabilities ÷ assets",
        ORANGE if row_focus.get("gearing", 0) > 100 else AMBER)

    st.markdown("")
    left, right = st.columns(2)

    with left:
        plot = blank_years(ent[["year", "total_assets", "total_liabilities",
                                "total_equity"]])
        fig = go.Figure()
        fig.add_bar(x=plot["year"], y=plot["total_assets"], name="Total assets",
                    marker_color=CYAN, opacity=.85)
        fig.add_bar(x=plot["year"], y=plot["total_liabilities"],
                    name="Total liabilities", marker_color=ORANGE, opacity=.75)
        fig.add_scatter(x=plot["year"], y=plot["total_equity"], name="Total equity",
                        mode="lines+markers", connectgaps=False,
                        line=dict(color=GOLD, width=2.6))
        fig.add_hline(y=0, line_color=MUTED, line_width=1)
        fig.update_layout(title="Balance-sheet insolvency", barmode="group",
                          yaxis=dict(title="R million"))
        period_bands(fig, ent[["year", "period"]])
        show(fig)

    with right:
        plot = blank_years(ent[["year", "accumulated_loss", "share_capital",
                                "shareholder_contribution"]])
        fig = go.Figure()
        fig.add_bar(x=plot["year"], y=plot["accumulated_loss"],
                    name="Accumulated loss", marker_color=ORANGE)
        fig.add_bar(x=plot["year"], y=plot["share_capital"], name="Share capital",
                    marker_color=CYAN)
        if plot["shareholder_contribution"].notna().any():
            fig.add_bar(x=plot["year"], y=plot["shareholder_contribution"],
                        name="Shareholder contribution", marker_color=GOLD)
        fig.update_layout(title="Capital injected against losses absorbed",
                          barmode="relative", yaxis=dict(title="R million"))
        show(fig)

    left, right = st.columns(2)
    with left:
        plot = blank_years(ent[["year", "current_ratio", "quick_ratio", "cash_ratio"]])
        fig = go.Figure()
        for column, name, colour in [("current_ratio", "Current ratio", CYAN),
                                     ("quick_ratio", "Quick ratio", AMBER),
                                     ("cash_ratio", "Cash ratio", VIOLET)]:
            fig.add_scatter(x=plot["year"], y=plot[column], name=name,
                            mode="lines+markers", connectgaps=False,
                            line=dict(color=colour, width=2.4))
        fig.add_hline(y=1, line_color=TEAL, line_dash="dash", line_width=1.4,
                      annotation_text="Solvency benchmark 1.0",
                      annotation_font=dict(size=10, color=TEAL))
        fig.update_layout(title="Liquidity", yaxis=dict(title="Ratio"))
        show(fig)

    with right:
        plot = blank_years(ent[["year", "working_capital", "debt"]])
        fig = go.Figure()
        fig.add_bar(x=plot["year"], y=plot["working_capital"],
                    name="Working capital", marker_color=ORANGE)
        fig.add_scatter(x=plot["year"], y=plot["debt"], name="Interest-bearing debt",
                        mode="lines+markers", connectgaps=False,
                        line=dict(color=GOLD, width=2.4))
        fig.add_hline(y=0, line_color=MUTED, line_width=1)
        fig.update_layout(title="Working capital deficit and debt stack",
                          yaxis=dict(title="R million"))
        show(fig)

    with st.expander("Interpretation — what this view shows"):
        neg_equity = ent[ent["total_equity"] < 0]["year"]
        wc = series["working_capital"].dropna()
        if neg_equity.empty or wc.empty:
            st.markdown("- Insufficient data in the selected range.")
        else:
            st.markdown(
                f"- Equity turns negative from FY{int(neg_equity.min())} and stays negative: "
                f"liabilities exceed assets, which is technical insolvency and the trigger "
                f"discussed in the going-concern and business-rescue analysis.\n"
                f"- The working-capital deficit widens to {rm(wc.iloc[-1])} by "
                f"FY{int(wc.index[-1])}, so short-term obligations could only be met from "
                f"new borrowing or shareholder support, not from operations.\n"
                f"- Capital injections are visible as gold bars, but they are absorbed by the "
                f"accumulated loss rather than rebuilding the equity base."
            )

# --------------------------------------------------------------------------
# TAB 4 — Cash and funding
# --------------------------------------------------------------------------
with tab_cash:
    left, right = st.columns([3, 2])

    with left:
        plot = blank_years(ent[["year", "net_operating_cash", "net_investing_cash",
                                "net_financing_cash"]])
        fig = go.Figure()
        fig.add_bar(x=plot["year"], y=plot["net_operating_cash"], name="Operating",
                    marker_color=ORANGE)
        fig.add_bar(x=plot["year"], y=plot["net_investing_cash"], name="Investing",
                    marker_color=VIOLET)
        fig.add_bar(x=plot["year"], y=plot["net_financing_cash"], name="Financing",
                    marker_color=CYAN)
        fig.add_hline(y=0, line_color=MUTED, line_width=1)
        fig.update_layout(title="Cash flow by activity", barmode="group",
                          yaxis=dict(title="R million"))
        period_bands(fig, ent[["year", "period"]])
        show(fig, 400)

    with right:
        row_focus = series.loc[focus] if focus in series.index else pd.Series(dtype=float)
        prior_years = [y for y in series.index if y < focus]
        opening = series.loc[max(prior_years), "cash"] if prior_years else np.nan
        closing = row_focus.get("cash", np.nan)
        flows = [row_focus.get(c, np.nan) for c in
                 ("net_operating_cash", "net_investing_cash", "net_financing_cash")]
        # Residual is the foreign-exchange effect on cash held offshore.
        fx = closing - (opening + np.nansum(flows)) if pd.notna(closing) else np.nan
        steps = [
            ("Opening cash", opening, "absolute"),
            ("Operating", flows[0], "relative"),
            ("Investing", flows[1], "relative"),
            ("Financing", flows[2], "relative"),
            ("FX effect", fx, "relative"),
            ("Closing cash", closing, "total"),
        ]
        fig = go.Figure(go.Waterfall(
            orientation="v",
            measure=[s[2] for s in steps],
            x=[s[0] for s in steps],
            y=[s[1] for s in steps],
            connector=dict(line=dict(color=LINE)),
            increasing=dict(marker=dict(color=TEAL)),
            decreasing=dict(marker=dict(color=ORANGE)),
            totals=dict(marker=dict(color=CYAN)),
            textposition="outside", texttemplate="%{y:,.0f}",
            textfont=dict(family="IBM Plex Mono, monospace", size=10),
        ))
        fig.update_layout(title=f"FY{focus} cash bridge",
                          yaxis=dict(title="R million"))
        show(fig, 400)

    left, right = st.columns(2)
    with left:
        columns = [c for c in ["shareholder_bailout", "borrowings_raised",
                               "net_operating_cash"] if c in ent.columns]
        plot = blank_years(ent[["year"] + columns])
        fig = go.Figure()
        if "shareholder_bailout" in columns:
            fig.add_bar(x=plot["year"], y=plot["shareholder_bailout"],
                        name="Shareholder contribution", marker_color=GOLD)
        if "borrowings_raised" in columns:
            fig.add_bar(x=plot["year"], y=plot["borrowings_raised"],
                        name="Borrowings raised", marker_color=CYAN)
        fig.add_scatter(x=plot["year"], y=plot["net_operating_cash"].abs(),
                        name="Operating cash burn", mode="lines+markers",
                        connectgaps=False, line=dict(color=ORANGE, width=2.4))
        fig.update_layout(title="Funding the burn: bailouts and borrowings",
                          barmode="stack", yaxis=dict(title="R million"))
        show(fig)

    with right:
        plot = blank_years(ent[["year", "capex", "depreciation", "cash"]])
        fig = go.Figure()
        fig.add_bar(x=plot["year"], y=plot["capex"], name="Capital expenditure",
                    marker_color=CYAN, opacity=.85)
        fig.add_bar(x=plot["year"], y=plot["depreciation"], name="Depreciation",
                    marker_color=MUTED, opacity=.6)
        fig.add_scatter(x=plot["year"], y=plot["cash"], name="Closing cash",
                        mode="lines+markers", connectgaps=False,
                        line=dict(color=GOLD, width=2.4))
        fig.update_layout(title="Reinvestment against depreciation",
                          barmode="group", yaxis=dict(title="R million"))
        show(fig)

    with st.expander("Interpretation — what this view shows"):
        cfo = series["net_operating_cash"].dropna()
        bail = series.get("shareholder_bailout", pd.Series(dtype=float)).dropna()
        st.markdown(
            f"- Operating cash flow is negative in {(cfo < 0).sum()} of the "
            f"{len(cfo)} published years, so the airline consumed cash in the ordinary "
            f"course of business rather than generating it.\n"
            f"- Financing inflows are the only positive bar in most years: the cash "
            f"balance is sustained by borrowings and shareholder support, which is the "
            f"classic profile of a business that is solvent only while the guarantor "
            f"keeps paying.\n"
            + (f"- Shareholder contributions of {rm(bail.sum())} across "
               f"FY{int(bail.index.min())}–FY{int(bail.index.max())} arrive in the same "
               f"years as the largest operating outflows.\n" if len(bail) else "")
            + "- Capital expenditure runs at or below depreciation in the later years, "
              "which signals an ageing fleet being run for cash rather than renewed."
        )

# --------------------------------------------------------------------------
# TAB 5 — Distress and red flags (the signature view)
# --------------------------------------------------------------------------
with tab_distress:
    st.markdown("### Descent profile — Altman Z''-score")
    plot = blank_years(ent[["year", "altman_z"]])
    fig = go.Figure()
    fig.add_hrect(y0=5.85, y1=max(12, np.nanmax(plot["altman_z"]) + 2),
                  fillcolor=TEAL, opacity=.10, line_width=0,
                  annotation_text="SAFE ZONE  Z'' > 5.85", annotation_position="top left",
                  annotation_font=dict(family="Barlow Condensed", size=11, color=TEAL))
    fig.add_hrect(y0=4.15, y1=5.85, fillcolor=AMBER, opacity=.12, line_width=0,
                  annotation_text="GREY ZONE", annotation_position="top left",
                  annotation_font=dict(family="Barlow Condensed", size=11, color=AMBER))
    fig.add_hrect(y0=min(-30, np.nanmin(plot["altman_z"]) - 3), y1=4.15,
                  fillcolor=ORANGE, opacity=.13, line_width=0,
                  annotation_text="DISTRESS ZONE  Z'' < 4.15",
                  annotation_position="bottom left",
                  annotation_font=dict(family="Barlow Condensed", size=11, color=ORANGE))
    fig.add_scatter(x=plot["year"], y=plot["altman_z"], mode="lines+markers+text",
                    name="Altman Z''", connectgaps=False,
                    line=dict(color=TEXT, width=3), marker=dict(size=9, color=GOLD),
                    text=[f"{v:,.1f}" if pd.notna(v) else "" for v in plot["altman_z"]],
                    textposition="top center",
                    textfont=dict(family="IBM Plex Mono, monospace",
                                  size=10, color=MUTED))
    fig.update_layout(title="Emerging-market Z''-score: the airline flying into terrain",
                      yaxis=dict(title="Z''-score"), showlegend=False)
    show(fig, 430)

    left, right = st.columns([2, 3])

    with left:
        components = blank_years(ent[["year", "z_x1_liquidity", "z_x2_retained",
                                      "z_x3_earnings", "z_x4_solvency"]])
        fig = go.Figure()
        for column, name, colour in [
            ("z_x1_liquidity", "X₁ working capital / assets", CYAN),
            ("z_x2_retained", "X₂ accumulated loss / assets", ORANGE),
            ("z_x3_earnings", "X₃ operating result / assets", AMBER),
            ("z_x4_solvency", "X₄ equity / liabilities", VIOLET),
        ]:
            fig.add_scatter(x=components["year"], y=components[column], name=name,
                            mode="lines", connectgaps=False,
                            line=dict(color=colour, width=2.2))
        fig.add_hline(y=0, line_color=MUTED, line_width=1)
        fig.update_layout(title="Which component drives the score")
        show(fig, 360)

    with right:
        flags = flag_matrix(ent)
        heat = flags.astype(int).T
        fig = go.Figure(go.Heatmap(
            z=heat.values, x=[f"FY{y}" for y in heat.columns], y=heat.index,
            colorscale=[[0, SURFACE], [1, ORANGE]], showscale=False,
            xgap=3, ygap=3,
            hovertemplate="%{y}<br>%{x}: %{z}<extra></extra>",
        ))
        fig.update_layout(title="Red-flag scorecard (shaded = signal present)")
        fig.update_yaxes(tickfont=dict(size=11))
        show(fig, 360)

    left, right = st.columns(2)
    with left:
        counts = flags.sum(axis=1).reindex(YEARS)
        fig = go.Figure(go.Bar(
            x=counts.index, y=counts.values,
            marker_color=[ORANGE if v >= 6 else AMBER if v >= 3 else TEAL
                          for v in counts.fillna(0)],
            text=[int(v) if pd.notna(v) else "" for v in counts],
            textposition="outside",
            textfont=dict(family="IBM Plex Mono, monospace", size=11),
        ))
        fig.update_layout(title=f"Warning signals active per year (of {len(FLAG_RULES)})",
                          yaxis=dict(title="Signals"))
        show(fig, 340)

    with right:
        plot = blank_years(ent[["year", "accruals_ratio", "days_sales_outstanding",
                                "cash_conversion"]])
        fig = go.Figure()
        fig.add_bar(x=plot["year"], y=plot["accruals_ratio"],
                    name="Accruals ratio (%)", marker_color=VIOLET, opacity=.8)
        fig.add_scatter(x=plot["year"], y=plot["days_sales_outstanding"],
                        name="Days sales outstanding", yaxis="y2",
                        mode="lines+markers", connectgaps=False,
                        line=dict(color=AMBER, width=2.4))
        fig.update_layout(
            title="Earnings quality: accruals and receivables",
            yaxis=dict(title="%"),
            yaxis2=dict(title="Days", overlaying="y", side="right", showgrid=False),
        )
        show(fig, 340)

    if not restated.empty:
        st.markdown("### Restatements between published reports")
        view = restated.copy()
        cols = [c for c in ["entity", "statement", "year", "dedup_key",
                            "original_rm", "restated_rm", "difference_rm", "pct_change"]
                if c in view.columns]
        view = (view[cols]
                .reindex(view["difference_rm"].abs().sort_values(ascending=False).index)
                .head(15))
        table(view.style.format({"original_rm": "{:,.0f}", "restated_rm": "{:,.0f}",
                                 "difference_rm": "{:,.0f}", "pct_change": "{:,.1f}%"}))
        pass

    with st.expander("Interpretation — what this view shows"):
        z = series["altman_z"].dropna()
        flags = flag_matrix(ent)
        counts = flags.sum(axis=1)
        breach = counts[counts >= 5]
        if z.empty:
            st.markdown("- Insufficient data in the selected range.")
        else:
            st.markdown(
                f"- The Z''-score falls from {num(z.iloc[0], 2)} in FY{int(z.index[0])} to "
                f"{num(z.iloc[-1], 2)} in FY{int(z.index[-1])}, far below the 4.15 distress "
                f"threshold, so statistical failure prediction flagged this business years "
                f"before business rescue was applied for in December 2019.\n"
                + (f"- At least five of the ten warning signals were simultaneously active "
                   f"from FY{int(breach.index.min())} onward.\n" if len(breach) else "")
                + f"- Decomposition shows the score is driven by X₂ and X₄ — accumulated "
                  f"losses and negative equity — rather than by a single bad trading year, "
                  f"which is what distinguishes structural failure from a cyclical downturn."
            )

# --------------------------------------------------------------------------
# TAB 6 — Predictive analytics (ML outputs from saa_ml.py)
# --------------------------------------------------------------------------
with tab_ml:
    ml_meta, ml_scores, ml_importance, ml_roc = load_ml()

    if not ml_meta:
        st.info(
            "ML outputs not found. Generate them first:\n\n"
            "```\npython saa_ml.py --horizon 1 --arff-dir raw --outdir data\n```\n\n"
            "You need the Polish Companies Bankruptcy dataset (UCI, DOI 10.24432/C5F600) "
            "placed as `.arff` files inside the `raw/` folder."
        )
    else:
        model_names = list(ml_meta.get("models", {}).keys())
        lr_key = next((m for m in model_names if "logistic" in m.lower()), model_names[0])
        rf_key = next((m for m in model_names if "forest" in m.lower()), model_names[-1])
        lr_meta = ml_meta["models"][lr_key]
        rf_meta = ml_meta["models"][rf_key]
        horizon = ml_meta.get("horizon_years", 1)

        # ── KPI row ──────────────────────────────────────────────────────────
        focus_row = ml_scores[ml_scores["year"] == focus] if not ml_scores.empty else pd.DataFrame()
        lr_prob = focus_row[lr_key].values[0] * 100 if not focus_row.empty else np.nan
        rf_prob = focus_row[rf_key].values[0] * 100 if not focus_row.empty else np.nan
        best_auc = max(lr_meta["cv_roc_auc"], rf_meta["cv_roc_auc"])
        best_recall = max(lr_meta["tuned_recall"], rf_meta["tuned_recall"])
        n_train = ml_meta.get("training_rows", 0)
        n_fail = ml_meta.get("training_failures", 0)

        cards = st.columns(5)
        kpi(cards[0], f"Failure probability FY{focus} · {lr_key.split()[0]}",
            f"{lr_prob:.1f}%" if pd.notna(lr_prob) else "—",
            f"{horizon}-year horizon", ORANGE if lr_prob > 50 else TEAL)
        kpi(cards[1], f"Failure probability FY{focus} · {rf_key.split()[0]}",
            f"{rf_prob:.1f}%" if pd.notna(rf_prob) else "—",
            f"{horizon}-year horizon", ORANGE if rf_prob > 50 else TEAL)
        kpi(cards[2], "Best cross-validated AUC", f"{best_auc:.3f}",
            "5-fold stratified", CYAN)
        kpi(cards[3], "Best recall (tuned cut-off)",
            f"{best_recall:.3f}", "share of failures caught", AMBER)
        kpi(cards[4], "Training sample", f"{n_train:,}",
            f"{n_fail:,} failures", MUTED)

        st.markdown("")

        # ── SAA failure probability over time ────────────────────────────────
        st.markdown("### SAA failure probability, scored year by year")
        st.markdown(
            f"<div style='font-family:IBM Plex Mono,monospace;font-size:.82rem;"
            f"color:{MUTED};margin-bottom:.6rem'>Probability of failure within "
            f"one year (%)</div>",
            unsafe_allow_html=True,
        )
        fig = go.Figure()
        period_bands(fig, ent[ent["year"].isin(ml_scores["year"])][["year", "period"]])
        for col, name, colour in [
            (lr_key, "Logistic regression", ORANGE),
            (rf_key, "Random forest", CYAN),
        ]:
            if col not in ml_scores.columns:
                continue
            pct = ml_scores[col] * 100
            fig.add_scatter(
                x=ml_scores["year"], y=pct, name=name, mode="lines+markers+text",
                connectgaps=False, line=dict(color=colour, width=2.5),
                marker=dict(size=8),
                text=[f"{v:.0f}%" for v in pct],
                textposition="top center",
                textfont=dict(family="IBM Plex Mono, monospace", size=10, color=colour),
            )
        lr_cut = lr_meta["tuned_threshold"] * 100
        rf_cut = rf_meta["tuned_threshold"] * 100
        fig.add_hline(y=lr_cut, line_dash="dot", line_color=ORANGE, line_width=1.2,
                      annotation_text=f"Logistic cut-off {lr_cut:.0f}%",
                      annotation_position="bottom right",
                      annotation_font=dict(size=10, color=ORANGE))
        fig.add_hline(y=rf_cut, line_dash="dot", line_color=CYAN, line_width=1.2,
                      annotation_text=f"Random forest cut-off {rf_cut:.0f}%",
                      annotation_position="bottom right",
                      annotation_font=dict(size=10, color=CYAN))
        fig.update_layout(
            title="SAA failure probability vs one-year horizon cut-offs",
            yaxis=dict(title="%", ticksuffix="%", range=[0, 125]),
        )
        show(fig, 430)

        # ── ML vs Z-score  |  ROC curves ────────────────────────────────────
        left, right = st.columns(2)

        with left:
            merged = ml_scores.merge(
                ent[["year", "altman_z"]], on="year", how="left"
            )
            fig = go.Figure()
            if lr_key in merged.columns:
                fig.add_bar(
                    x=merged["year"], y=merged[lr_key] * 100,
                    name="Failure probability (logistic)", marker_color=ORANGE,
                )
            fig.add_scatter(
                x=merged["year"], y=merged["altman_z"], name="Altman Z''",
                yaxis="y2", mode="lines+markers", connectgaps=False,
                line=dict(color=GOLD, width=2.4), marker=dict(size=7),
            )
            fig.update_layout(
                title="Machine learning against the classical Z''-score",
                yaxis=dict(title="Probability %", ticksuffix="%"),
                yaxis2=dict(title="Z''-score", overlaying="y", side="right",
                            showgrid=False),
            )
            show(fig, 400)

        with right:
            if not ml_roc.empty:
                fig = go.Figure()
                colours_roc = {lr_key: ORANGE, rf_key: CYAN}
                for model_name in ml_roc["model"].unique():
                    subset = ml_roc[ml_roc["model"] == model_name]
                    auc = ml_meta["models"][model_name]["holdout_roc_auc"]
                    fig.add_scatter(
                        x=subset["fpr"], y=subset["tpr"],
                        name=f"{model_name} (AUC {auc:.3f})",
                        mode="lines",
                        line=dict(color=colours_roc.get(model_name, VIOLET), width=2.2),
                    )
                fig.add_scatter(
                    x=[0, 1], y=[0, 1], name="Random",
                    mode="lines", line=dict(color=MUTED, dash="dash", width=1),
                )
                fig.update_layout(
                    title="ROC curves on the held-out test set",
                    xaxis=dict(title="False positive rate"),
                    yaxis=dict(title="True positive rate"),
                )
                show(fig, 400)

        # ── Feature importance  |  Odds ratios ──────────────────────────────
        if not ml_importance.empty:
            left, right = st.columns(2)

            with left:
                imp = ml_importance.sort_values("rf_importance", ascending=True)
                fig = go.Figure(go.Bar(
                    x=imp["rf_importance"], y=imp["feature"], orientation="h",
                    marker_color=CYAN,
                    text=imp["rf_importance"].map(lambda v: f"{v:.3f}"),
                    textposition="outside",
                    textfont=dict(family="IBM Plex Mono, monospace", size=10),
                ))
                fig.update_layout(
                    title="Which ratios drive the prediction (random forest)",
                    xaxis=dict(title="Mean decrease in impurity"),
                )
                show(fig, 400)

            with right:
                st.markdown("### Odds ratios")
                odds = (ml_importance[["feature", "definition", "lr_odds_ratio_per_sd"]]
                        .sort_values("lr_odds_ratio_per_sd", ascending=False)
                        .reset_index(drop=True))
                table(odds.style.format({"lr_odds_ratio_per_sd": "{:.3f}"}))
                st.caption(
                    "Odds multiplier per one standard-deviation increase in the ratio, "
                    "holding all others constant. Values above 1.0 increase failure risk; "
                    "below 1.0 reduce it."
                )

        with st.expander("Interpretation — what this view shows"):
            st.markdown(
                f"- Models trained on {n_train:,} firm-years ({n_fail:,} failures) from "
                f"the Polish Companies Bankruptcy benchmark dataset, then scored against "
                f"SAA's own published ratios as an out-of-sample case study.\n"
                f"- Logistic regression (AUC {lr_meta['cv_roc_auc']:.3f}) and random "
                f"forest (AUC {rf_meta['cv_roc_auc']:.3f}) both place SAA in the "
                f"high-failure zone from the early years of the decade.\n"
                f"- The Youden-J tuned cut-off maximises the separation between true "
                f"positives and false alarms; at that threshold the best model catches "
                f"{best_recall:.1%} of actual failures in the held-out test set.\n"
                f"- The feature-importance panel shows which ratios are most predictive "
                f"across the benchmark population; the odds-ratio table shows their "
                f"direction in the logistic model."
            )

# --------------------------------------------------------------------------
# TAB 7 — Data and export
# --------------------------------------------------------------------------
with tab_data:
    st.markdown("### Parsed dataset")
    statements = st.multiselect(
        "Statement", sorted(tidy["statement"].unique()),
        default=sorted(tidy["statement"].unique()),
    )
    view = tidy[(tidy["entity"] == entity) & (tidy["statement"].isin(statements))
                & (tidy["year"].between(*year_range))]
    pivot = (view.pivot_table(index=["statement", "line_item"], columns="year",
                              values="value_rm", aggfunc="first")
             .rename(columns=lambda c: f"FY{c}" if isinstance(c, (int, float)) else c)
             .reset_index())
    table(pivot, height=420)

    st.markdown("### Ratio and indicator table")
    ratio_cols = ["year", "period", "ebitda_margin", "operating_margin", "net_margin",
                  "cost_to_income", "current_ratio", "quick_ratio", "gearing",
                  "equity_ratio", "interest_cover", "days_sales_outstanding",
                  "accruals_ratio", "cash_conversion", "altman_z"]
    ratios = ent[[c for c in ratio_cols if c in ent.columns]]
    table(ratios.style.format(precision=2))

    downloads = st.columns(3)
    downloads[0].download_button(
        "Download tidy dataset (CSV)", tidy.to_csv(index=False).encode(),
        "saa_financials_tidy.csv", "text/csv",
    )
    downloads[1].download_button(
        "Download ratios (CSV)", ratios.to_csv(index=False).encode(),
        f"saa_ratios_{entity.lower()}.csv", "text/csv",
    )
    if not restated.empty:
        downloads[2].download_button(
            "Download restatements (CSV)", restated.to_csv(index=False).encode(),
            "saa_restatements.csv", "text/csv",
        )

    st.markdown("### Method notes")
    st.markdown(
        f"""
- **Source.** SAA annual financial statements, parsed by `saa_etl.py` into
  {len(tidy):,} line-item values covering FY2010–FY2019 for both Group and Company.
- **Missing years.** FY2013 and FY2014 are not in the source workbook. Series break
  at that point; no value is interpolated.
- **Restated comparatives.** Where FY2017 appears in two reports, the later restated
  figure is used and the difference is retained for the restatement table.
- **Sign convention.** Costs are stored as magnitudes; results (EBITDA, operating
  result, loss for the year, equity) keep their published sign, so negative means loss.
- **Altman Z''.** Emerging-market variant for non-manufacturers:
  Z'' = 3.25 + 6.56·X₁ + 3.26·X₂ + 6.72·X₃ + 1.05·X₄, with X₁ = working capital/assets,
  X₂ = accumulated loss/assets, X₃ = operating result/assets, X₄ = equity/liabilities.
  Distress below 4.15, grey 4.15–5.85, safe above 5.85.
- **Periods.** Pre-scandal, scandal and collapse boundaries are set in the sidebar so
  the analysis can be re-run against whichever timeline the report defends.
"""
    )

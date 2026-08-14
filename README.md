# SAA Financial Intelligence Dashboard

A Power BI-style business-intelligence dashboard built entirely in Python
(Streamlit + Plotly), for **Question 4** of the integrated assessment —
South African Airways SOC Ltd and its associated companies.

Two files do the work:

| File | Role |
|---|---|
| `saa_etl.py` | Parses `SAA_Financial_statement.xlsx` into a tidy analytical dataset |
| `saa_dashboard.py` | Six-tab interactive dashboard with slicers, KPI scorecards, waterfalls, ratio trends and a red-flag engine |

---

## Quick start

```bash
pip install -r requirements.txt

# 1. Build the dataset (already built in data/, re-run if the workbook changes)
python saa_etl.py --source SAA_Financial_statement.xlsx --outdir data

# 2. Launch the dashboard
streamlit run saa_dashboard.py
```

It opens at `http://localhost:8501`. Nothing else is required — no Power BI
licence, no database.

---

## What the ETL produces

`saa_etl.py` reads all nine statement sheets and flattens 1 223 published
line-item values into long format:

```
statement | entity | year | line_item | metric | value_rm
```

Outputs in `data/`:

- `saa_financials_tidy.csv` — every parsed line item, Group and Company, FY2010–FY2019
- `saa_metrics_wide.csv` — 60 canonical metrics per entity-year, ready for ratio work
- `saa_restatements.csv` — line items whose FY2017 comparatives changed between the two published reports

Three things it handles that a manual copy-paste would not:

1. **Changing published wording.** FY2012 says "Revenue"/"Turnover"; FY2019 says
   "Total income"/"Airline revenue". Aliases fold these onto one metric so a
   ten-year trend is actually comparable.
2. **Accounting conventions.** `(558)`, `26 992`, `–` and real numbers are all
   parsed correctly; unlabelled subtotal rows are attributed to their section.
3. **Integrity check.** After parsing, assets = equity + liabilities is verified
   for all 16 entity-years. All 16 balance — the console prints this, and it is
   worth stating in your methodology paragraph.

---

## The dashboard, tab by tab

| Tab | Contents | Assessment link |
|---|---|---|
| **Overview** | KPI scorecard, income vs cost base vs bottom line, income-to-loss waterfall, period comparison | 4.2 KPI scorecards |
| **Profitability** | Margin trends, cost structure (100% stacked), income vs cost growth index, cost drivers | 4.1 performance trends |
| **Financial position** | Assets vs liabilities vs equity, capital injected against losses absorbed, liquidity ratios, working-capital deficit and debt stack | 4.1 financial distress indicators |
| **Cash & funding** | Cash flow by activity, cash bridge waterfall, bailouts and borrowings against the burn, capex vs depreciation | 4.1 cash-based distress |
| **Distress & red flags** | Altman Z''-score descent with distress zones, Z-component decomposition, ten-rule red-flag scorecard, accruals ratio and DSO, restatement table | 4.1 earnings manipulation and distress prediction |
| **Data** | Filterable statement pivot, ratio table, CSV exports, method notes | Appendix evidence |

**Slicers** (sidebar): reporting entity (Group / Company), financial-year range,
focus year, and — importantly — the scandal-period boundaries. Move the sliders
and every period band, average and comparison recomputes, so you can defend
whichever timeline your report argues for rather than hard-coding one.

Each tab has an **Interpretation** expander that writes its findings in
sentences generated from the data. Those are drafting prompts for 4.2, not
finished report prose — rewrite them in your own words.

---

## Headline findings (Group, R million)

| FY | Net margin % | Cost-to-income % | Current ratio | Gearing % | Interest cover | Z''-score |
|---|---|---|---|---|---|---|
| 2011 | 3.45 | 95.6 | 0.82 | 96.6 | 5.99 | −0.41 |
| 2012 | −3.53 | 105.5 | 0.68 | 97.1 | −8.95 | −1.96 |
| 2015 | −20.14 | 109.6 | 0.41 | 173.3 | −4.60 | −9.76 |
| 2016 | −4.81 | 98.3 | 0.55 | 171.6 | 0.51 | −5.45 |
| 2017 | −17.67 | 108.6 | 0.36 | 212.5 | −1.62 | −11.26 |
| 2018 | −18.65 | 109.1 | 0.29 | 199.5 | −1.81 | −16.32 |
| 2019 | −24.02 | 110.9 | 0.24 | 208.8 | −2.06 | −19.76 |

Points worth arguing in the report:

- **FY2011 is the last profitable year** (R779m profit); FY2012 is the first
  loss (R843m). Every published year after that is loss-making.
- **The current ratio never reaches 1.0 in the entire decade** and falls to 0.24
  by FY2019 — the warning was available to any reader of the statements from
  FY2010, long before business rescue in December 2019.
- **Liabilities exceed assets from FY2015** (gearing 173% rising to 209%), i.e.
  technical insolvency, consistent with the R17.802 billion figure your draft
  already cites for 31 March 2017.
- **The Altman Z''-score is in the distress zone in every year with data** and
  falls from −0.41 to −19.76. Decomposition shows accumulated losses (X₂) and
  negative equity (X₄) drive it, not one bad trading year.
- **R15 000m of shareholder contributions** (FY2018 R10 000m, FY2019 R5 000m)
  arrive in the same years as the largest operating cash outflows — the airline
  was solvent only while the guarantor kept paying.
- **FY2017 restatement.** Group trade and other receivables were restated from
  R5 333m to R3 976m between the two published reports — a R1 357m reduction,
  with a corresponding R945m reduction in current assets and R915m increase in
  non-current assets. That is a reclassification of over a billion rand in the
  comparatives of the year at the centre of the scandal, and it belongs in your
  Question 2.2 argument about the reliability of the financial statements.

---

## Method notes (put these in your methodology paragraph)

- All figures R million as published; no inflation adjustment.
- **FY2013 and FY2014 are not in the source workbook.** Every series breaks at
  that point; nothing is interpolated. Flag the gap as a limitation.
- Where FY2017 appears twice, the later restated figure is used; the difference
  is preserved in `saa_restatements.csv`.
- Costs are stored as magnitudes; results (EBITDA, operating result, loss for
  the year, equity) keep their published sign, so negative means loss.
- Altman Z''-score, emerging-market variant for non-manufacturers:
  `Z'' = 3.25 + 6.56·X₁ + 3.26·X₂ + 6.72·X₃ + 1.05·X₄`, distress below 4.15,
  grey 4.15–5.85, safe above 5.85. X₂ uses accumulated loss in place of
  retained earnings and X₃ uses the operating result as the EBIT proxy — state
  this, since a marker may check it.
- The red-flag engine applies ten rules per year (`FLAG_RULES` in
  `saa_dashboard.py`); the rules are visible in the code and easy to defend.

---

## Using this in the submission

- **Screenshots.** `figures/fig1_overview.png` … `fig5_distress.png` are
  full-page captures of each tab, ready to drop into the report as figures
  under 4.2 with your own interpretation beneath each one. Re-capture after
  changing any slicer so the figure matches what you describe.
- **Supporting analytical code.** The brief asks for it — submit `saa_etl.py`
  and `saa_dashboard.py` as the Python component, and mention that the ETL is
  reproducible from the raw workbook.
- **Financial analysis workbook.** `data/saa_metrics_wide.csv` opens directly in
  Excel and is a clean base for the dynamic-reference workbook required by 4.1.
- **Live demo.** If your group presents, run the app and move the scandal-period
  sliders — interactivity is what separates a dashboard from a picture of a chart.

---

## Not covered here

This handles 4.1 and 4.2. Still outstanding for the rest of Question 4:

- **4.3** three-statement five-year forecast with a counterfactual "no scandal" scenario
- **4.4** two machine-learning models for share-price forecasting (note: SAA is
  state-owned and unlisted, so there is no share price — you will need to argue
  a proxy, such as a listed comparator or a bond-spread/valuation proxy, and
  justify it explicitly)
- **4.5** DCF, DDM and relative-multiple valuation with sensitivity analysis

The tidy dataset is the natural input for all three.

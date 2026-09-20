# Commercial-Banking-Fraud-Mule-Account-and-Transaction-Laundering-Analytics
MuleWatch -  Capstone Project : behavioural, graph &amp; NLP analytics for banking fraud, mule-account and transaction-laundering detection, on real (IBM AML) + synthetic data.

# T17 · MuleWatch

**Behavioural, Graph and NLP Analytics for Mule-Account and Transaction-Laundering Detection**

SAS821S Security Analytics Capstone, NUST — a fictional retail bank, "MetroTrust Bank."

| | |
|---|---|
| **Member A** | Romario Rhoman (221080457) — Data and Modelling Lead |
| **Member B** | Vaino Uusiku (217150802) — Security Engineering and Intelligence Lead |
| **Facilitator** | Prof. Attlee M. Gamundani |

## What this project does

MuleWatch detects three fraud typologies for a fictional bank's digital banking channel:
account takeover, mule-account networks, and transaction laundering. It combines a
supervised transaction classifier, unsupervised access-anomaly detection, graph-based
account-cluster analysis, and NLP-based AML case-note classification into a single
ranked, explainable alert queue for analyst triage.

## Data

The transaction and account-graph backbone uses **real, public data**: IBM's Anti-Money-
Laundering transaction dataset (Kaggle, `HI-Small` split), licensed under CDLA-Sharing-1.0.

> Altman, E. et al. (2023). *Realistic Synthetic Financial Transactions for Anti-Money
> Laundering Models.* NeurIPS 2023 Datasets and Benchmarks Track.

Auth/MFA logs, device fingerprints, beneficiary-change history, KYC attributes and AML
case notes have no public real-world equivalent and are synthetically generated, keyed
to the real account IDs from the dataset above.

**Before running the notebook**, download both files from
[Kaggle](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml)
and place them in the repository root:

- `HI-Small_Trans.csv`
- `Small_Patterns.txt`

These files are **not committed to this repository** (large, and redistribution should
respect the dataset's own licence terms) — you must download them yourself.

## Requirements

```bash
pip install pandas numpy scikit-learn xgboost networkx python-louvain shap spacy matplotlib joblib
python -m spacy download en_core_web_sm
```

## Running it

1. Install the requirements above.
2. Download the two data files (see **Data**) into the repository root.
3. Open `MuleWatch_T17_Implementation.ipynb` and run all cells top to bottom.
   - The first cell loads a 5M-row CSV — this step alone takes a few minutes.
   - Total runtime end-to-end is roughly 10–15 minutes.
4. Generated data, models, charts and the analyst dashboard are written to a
   `T17_MuleWatch/` folder created alongside the notebook (see **Repository structure**
   below) — this folder is regenerated on every run and is not committed to the
   repository.

## Repository structure

```
.
├── MuleWatch_T17_Implementation.ipynb   # main notebook — the full pipeline
├── README.md
├── SAS821S_T17_Project_Charter.docx     # project charter
├── SAS821S_T17_Implementation_Plan.docx # Milestone 2 deliverable
└── T17_MuleWatch/                       # generated on each run — not committed
    ├── 02_data/{raw,processed}/         # cleaned data, feature tables, graph edges
    ├── 04_models/                       # persisted model artefacts + model_card.json
    ├── 05_simulation/                   # Monte Carlo threshold-sensitivity results
    ├── 06_text_mining/                  # processed case notes, extracted entities
    ├── 07_dashboard_or_prototype/       # app.py (Streamlit analyst dashboard)
    └── 08_outputs/                      # charts, reports, intelligence products
```

## Running the analyst dashboard

After the notebook has run at least once (so `T17_MuleWatch/07_dashboard_or_prototype/`
contains the generated `app.py` and CSV snapshot):

```bash
cd T17_MuleWatch/07_dashboard_or_prototype
streamlit run app.py
```


## License

Project code: for academic submission as part of SAS821S at NUST. The IBM AML dataset
used for the transaction/graph backbone is licensed separately under CDLA-Sharing-1.0 by
its original publisher and is not redistributed in this repository.

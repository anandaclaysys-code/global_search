# SCCU DigitalBankingNavBot — Intent Classification
### Project Summary Document

---

## 1. What This Project Does

The SCCU bot helps users **navigate a digital banking app** by understanding natural language queries (e.g. *"pay my bill"*, *"nearest ATM"*) and routing them to the correct feature screen.

Two engines run in parallel:

| Engine | Method | Best At |
|---|---|---|
| Inverted Index | Word-match ranking | Fast keyword lookups |
| TF-IDF + Logistic Regression | ML classification | Probabilistic intent prediction |

**Source data:** `SCCU-test.json` — a Microsoft Azure CLU (Conversational Language Understanding) project export.

---

## 2. Dataset at a Glance

| Property | Value |
|---|---|
| Source | Azure CLU export (`2025-05-15-preview`) |
| Total utterances | 451 |
| After cleaning & deduplication | **387 usable rows** |
| Unique intents | **43** |
| Language | English (`en-us`) |
| Avg. examples per intent | ~9 |

**Intent categories (key ones):**
`Accounts`, `ApplyLoan`, `ATMLocator`, `BillPay`, `CardControls`, `CheckDeposit`, `CrossAccountTransfers`, `Dashboard`, `ExternalACHAccount`, `LoanPayOff`, `LoanServices`, `Message`, `MyOffers`, `MyProfile`, `NoticesandStatements`, `OpenAccount`, `PaymentCalender`, `TransferMoney`, `TravelPlans`, `VisitaBranch` … *(43 total)*

**Entity types used for keyword matching:**
- `navigationkeyword` — 40 navigation destinations with synonyms
- `loankeyword` — loan product keywords (Auto, Personal, Home Equity, Boat/RV…)
- `livechatkeyword` — live agent trigger phrases

---

## 3. Pipeline

```
SCCU-test.json
  ↓  extract utterances (text + intent)
DataFrame (451 rows)
  ↓  lowercase → strip whitespace → drop special-char rows → deduplicate
  ↓  remove stop words → Porter stemming
Clean DataFrame (387 rows, 43 intents)
  ├──→ Inverted Index  →  ranked match-count results
  └──→ TF-IDF + Logistic Regression
          ↓  80/20 stratified split  (train=309, test=78, seed=42)
          ↓  TfidfVectorizer (unigrams + bigrams, sublinear_tf=True)
          ↓  LogisticRegression (C=5, max_iter=1000)
          ↓  classification_report + accuracy
          ↓  export → SCCU_tfidf_results.xlsx  (3 sheets)
```

**Key preprocessing choices:**
- Custom stop-word list (40 words) — intentionally keeps *"pay"*, *"my"* as they are meaningful in banking
- Porter stemming: `"transfers"` → `"transfer"`, `"statements"` → `"statement"`
- Bigrams capture: `"bill pay"`, `"check deposit"`, `"loan payoff"` as distinct features

---

## 4. Model Performance

**Overall accuracy: 87.18%** on 78 test samples

| Metric | Score |
|---|---|
| Macro Precision | 0.93 |
| Macro Recall | 0.88 |
| **Macro F1** | **0.88** |
| Weighted F1 | 0.87 |

**Well-performing intents (F1 = 1.00):**
`AccountInformation`, `BoatRVLoan`, `CardControls`, `CrossAccountTransfers`, `PaymentCalender`, `TransferMoney`, `TravelPlans`

**Problem intents:**

| Intent | F1 | Issue |
|---|---|---|
| `LoanServices` | 0.00 | Ambiguous with all other loan intents; too few examples |
| `MyProfile` | 0.53 | Broad vocab (*email, phone, contact*) causes false positives |
| `CheckDeposit` | 0.50 | Unseen phrasings in test; very low training support |

---

## 5. Output Files

| File | Description |
|---|---|
| `SCCU-test.json` | Source CLU export — all intents, entities, utterances |
| `tfidf_intent_classification.ipynb` | Notebook with full pipeline code |
| `SCCU_tfidf_results.xlsx` | Excel output with 3 sheets: `all`, `train`, `test_predictions` |
| `requirements.txt` | Python dependencies |

**`test_predictions` sheet columns:** `text`, `intent` (true), `predicted`, `correct` (True/False)

---

## 6. Known Issues

| Issue | Impact |
|---|---|
| ~9 examples per intent on average | Model fails on rare/ambiguous intents |
| Single 80/20 split (78 test samples) | Accuracy has ±5% variance — not statistically reliable |
| TF-IDF cannot handle paraphrases | *"move funds"* ≠ *"transfer money"* at the token level |
| No confidence threshold gate | Low-confidence predictions are returned without warning |
| Inverted Index uses raw word counts | Common words like *"pay"* overpower rare discriminative words |

---

## 7. Recommended Next Steps

1. **Add cross-validation** (5-fold stratified) — current single split is unreliable
2. **Add Top-3 accuracy & MRR** — the system ranks Top-5 but only Top-1 is evaluated
3. **Augment data to ≥20 utterances per intent** — highest-leverage improvement available
4. **Replace LogReg with LinearSVC** — better for sparse high-dim text; minimal code change
5. **Replace raw Inverted Index with BM25+** — `pip install rank-bm25`; proper IDF weighting
6. **Add confidence threshold gating** — abstain when `max_proba < 0.70` (matches CLU config)

---

## 8. How to Run

```powershell
# Activate environment
.venv\Scripts\Activate.ps1

# Install dependencies
pip install scikit-learn pandas matplotlib seaborn openpyxl nltk rank-bm25

# Launch notebook
jupyter notebook tfidf_intent_classification.ipynb
# Then: Kernel → Restart & Run All
```

> **Note:** Run the notebook from the project folder — `SCCU-test.json` is referenced by relative path.

---

*SCCU DigitalBankingNavBot | ClaySys Technologies | June 2026*

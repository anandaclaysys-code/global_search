# SCCU Intent Classification — Current Implementation

> Documents **exactly what is built today** in `tfidf_intent_classification.ipynb`.

---

## 1. Environment

| Item | Detail |
|---|---|
| Language | Python 3.14.5 |
| Runtime | Jupyter Notebook |
| Virtual env | `.venv` (local) |
| Kernel | `.venv` |

**Installed packages (`requirements.txt`):**
```
json · sklearn · pandas · matplotlib · seaborn · openpyxl · nltk
```

---

## 2. Input Data

**File:** `SCCU-test.json`  
**Format:** Azure CLU project export (`2025-05-15-preview`)  
**Fields used:** `assets.utterances` → `text`, `intent` only  
**Fields ignored:** `entities`, `dataset`, `language` (per utterance)

---

## 3. Data Loading

```python
with open('SCCU-test.json', encoding='utf-8') as f:
    raw = json.load(f)

utts = raw['assets']['utterances']
df = pd.DataFrame([{'text': u['text'], 'intent': u['intent']} for u in utts])
# Result: (451, 2)
```

---

## 4. Preprocessing

Applied in this exact order:

```python
# Step 1 — Normalise
df['text'] = df['text'].str.strip().str.lower()
df['text'] = df['text'].apply(lambda x: re.sub(r'\s+', ' ', x))

# Step 2 — Drop rows with no alphanumeric content
mask = df['text'].str.contains(r'[a-z0-9]', regex=True)
df = df[mask].copy()                         # removed 37 rows (SpecialCharacters intent)

# Step 3 — Drop duplicates
df = df.drop_duplicates(subset=['text', 'intent']).reset_index(drop=True)

# Step 4 — Stop word removal
stop_words = {
    'the','a','an','and','or','but','in','on','at','to','for',
    'of','with','is','was','are','been','be','have','has','had',
    'do','does','did','will','would','could','should','may','might',
    'must','can','this','that','these','those','i','you','he','she',
    'it','we','they','what','which','who','when','where','why','how'
}
df['text'] = df['text'].apply(
    lambda t: ' '.join(w for w in t.split() if w not in stop_words)
)

# Step 5 — Porter stemming
stemmer = PorterStemmer()
df['text'] = df['text'].apply(
    lambda t: ' '.join(stemmer.stem(w) for w in t.split())
)
```

**Result after preprocessing:**

| Stage | Rows |
|---|---|
| Raw | 451 |
| After dropping special-char rows | 414 |
| After deduplication | **387** |
| Unique intents | **43** |

---

## 5. Engine 1 — Inverted Index

**Built on the preprocessed `df` (387 rows).**

```python
inverted_index = defaultdict(list)
for idx, row in df.iterrows():
    for word in row['text'].split():
        if idx not in inverted_index[word]:
            inverted_index[word].append(idx)
# Total unique terms: 364
```

**Prediction function:**
```python
def predict_with_index(query):
    clean = stem_text(remove_stop_words(query.lower().strip()))
    intent_counts = defaultdict(int)
    for word in clean.split():
        if word in inverted_index:
            for doc_id in inverted_index[word]:
                intent_counts[df.iloc[doc_id]['intent']] += 1
    ranked = sorted(intent_counts.items(), key=lambda x: x[1], reverse=True)
    # Prints top 5 with match count and percentage
```

**What it does NOT do:** No IDF weighting. All word matches count equally.

---

## 6. Engine 2 — TF-IDF + Logistic Regression

### 6.1 Train/Test Split

```python
# Remove classes with only 1 sample (cannot stratify)
counts = df['intent'].value_counts()
df = df[df['intent'].isin(counts[counts >= 2].index)].reset_index(drop=True)

train, test = train_test_split(
    df, test_size=0.2, stratify=df['intent'], random_state=42
)
# train = 309 rows   test = 78 rows
```

### 6.2 Pipeline

```python
pipe = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 2),    # unigrams + bigrams
        sublinear_tf=True,     # log(1 + tf)
        norm=None              # no L2 normalisation
    )),
    ('clf', LogisticRegression(
        C=5,
        max_iter=1000,
        random_state=42
    ))
])

pipe.fit(train['text'], train['intent'])
preds = pipe.predict(test['text'])
```

### 6.3 Prediction Function

```python
def predict(query):
    clean = stem_text(remove_stop_words(query.lower().strip()))
    probs = pipe.predict_proba([clean])[0]
    top   = np.argsort(probs)[::-1][:5]
    for r, i in enumerate(top, 1):
        print(f'{r}. {pipe.named_steps["clf"].classes_[i]:<35} {probs[i]*100:.1f}%')
```

---

## 7. Evaluation (as currently implemented)

```python
print('accuracy:', round(accuracy_score(test['intent'], preds), 4))
print(classification_report(test['intent'], preds, zero_division=0))
```

**Results:**

| Metric | Value |
|---|---|
| Accuracy | **87.18%** |
| Macro Precision | 0.93 |
| Macro Recall | 0.88 |
| Macro F1 | 0.88 |
| Weighted F1 | 0.87 |
| Test samples | 78 |
| Intents in test | 35 |

**No cross-validation. No Top-K. No MRR. No calibration check.** Single split only.

---

## 8. Output Export

```python
train_out = train.assign(split='train')
test_out  = test.assign(
    split='test',
    predicted=preds,
    correct=(test['intent'].values == preds)
)

with pd.ExcelWriter('SCCU_tfidf_results.xlsx', engine='openpyxl') as w:
    df.to_excel(w, sheet_name='all', index=False)
    train_out.to_excel(w, sheet_name='train', index=False)
    test_out.to_excel(w, sheet_name='test_predictions', index=False)
```

**Output file:** `SCCU_tfidf_results.xlsx`

| Sheet | Columns |
|---|---|
| `all` | `text`, `intent` |
| `train` | `text`, `intent`, `split` |
| `test_predictions` | `text`, `intent`, `split`, `predicted`, `correct` |

---

## 9. Notebook Cell Order

| Cell | What it does |
|---|---|
| 1 | Imports |
| 2 | Load JSON → DataFrame (451 rows) |
| 3 | Define `stem_text()`, `remove_stop_words()`, stop words set |
| 4 | Build word frequency index (for preview only) |
| 5 | Build Inverted Index (364 terms) |
| 6 | `predict_with_index()` — test 3 sample queries |
| 7 | Full preprocessing → clean DataFrame (387 rows) |
| 8 | Train/test split (309/78) |
| 9 | Fit TF-IDF + LogReg pipeline → print accuracy |
| 10 | Print `classification_report` |
| 11 | `predict()` — test 3 sample queries with probabilities |
| 12 | Export to `SCCU_tfidf_results.xlsx` |

---

*SCCU DigitalBankingNavBot | Current Implementation | June 2026*

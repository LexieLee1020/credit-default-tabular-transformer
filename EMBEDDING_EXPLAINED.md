# embedding.py — Code Explanation

## What this file does
Takes the tokenizer output and converts each customer row into 24 tokens of 64 numbers each.
Output shape: **(B, 24, 64)** — B customers, 24 tokens, 64 numbers per token.

---

## Token order (positions 0–23)

| Position | Feature | Type |
|---|---|---|
| 0 | [CLS] | special learnable token |
| 1 | SEX | categorical |
| 2 | EDUCATION | categorical |
| 3 | MARRIAGE | categorical |
| 4 | PAY_0 | PAY hybrid |
| 5 | PAY_2 | PAY hybrid |
| 6 | PAY_3 | PAY hybrid |
| 7 | PAY_4 | PAY hybrid |
| 8 | PAY_5 | PAY hybrid |
| 9 | PAY_6 | PAY hybrid |
| 10 | LIMIT_BAL | numerical |
| 11 | AGE | numerical |
| 12–17 | BILL_AMT1–6 | numerical |
| 18–23 | PAY_AMT1–6 | numerical |

---

## The 4 embedding tables (`__init__`, lines 61–93)

### 1. Categorical embeddings (lines 61–64)
```python
self.cat_embeddings = nn.ModuleDict({
    "SEX":       nn.Embedding(2, 64),
    "EDUCATION": nn.Embedding(4, 64),
    "MARRIAGE":  nn.Embedding(3, 64),
})
```
Each feature has its own separate lookup table.
Give it an index (e.g. 1) → get back 64 numbers.
Vocab sizes (2, 4, 3) come from Phase 1 data cleaning.

**Why separate tables?** If SEX and EDUCATION shared one table, `SEX=1` and `EDUCATION=1` would land on the same row — the model would treat them as identical, which is wrong.

---

### 2. PAY state embedding (line 68)
```python
self.pay_state_embedding = nn.Embedding(4, 64)
```
4 possible states:
- 0 = no_bill (PAY = -2, no card use)
- 1 = paid_full (PAY = -1, paid in full)
- 2 = minimum (PAY = 0, minimum payment)
- 3 = delinquent (PAY = 1–8, months delayed)

### 2b. PAY severity projection (line 73)
```python
self.pay_severity_proj = nn.Linear(1, 64, bias=True)
```
Takes the severity float (e.g. PAY=3 → 3/8 = 0.375) and converts it to 64 numbers via:
```
output = 0.375 × [w1...w64] + [b1...b64]
```
Weights are learned during training. Bigger severity → bigger output → model knows PAY=6 is worse than PAY=2.

---

### 3. Numerical embeddings (lines 77–81)
```python
self.num_feature_embedding = nn.Embedding(14, 64)  # "which feature am I?"
self.value_proj = nn.Linear(1, 64, bias=True)       # "what is my value?"
```
Two parts combined:
- **Feature identity**: index 0=LIMIT_BAL, 1=AGE, 2=BILL_AMT1, ... tells the model *which* feature this is
- **Value projection**: takes the scaled number (e.g. 0.83 for credit limit) and converts to 64 numbers

The scaled number comes from StandardScaler applied in Phase 1 — raw 50,000 NT$ becomes ~0.83.

---

### 4. Positional embedding (line 85)
```python
self.pos_embedding = nn.Embedding(23, 64)
```
23 positions (one per feature token, not counting CLS).
Tells the model the order — position 0=SEX, position 6=PAY_3, etc.

### 4b. [CLS] token (line 88)
```python
self.cls_token = nn.Parameter(torch.randn(1, 1, 64) * 0.02)
```
A special learnable vector with no feature value. Prepended at position 0.
After self-attention, the transformer reads from [CLS] to make the final default/no-default prediction — it collects information from all 23 feature tokens via attention.

---

## Weight initialisation (lines 97–110)

Sets starting values before training begins:
- Embedding tables → small random numbers N(0, 0.02) — same as BERT
- Linear projections → Xavier normal — keeps gradients stable at the start
- Biases → zeros

---

## Forward pass — how tokens are built (lines 112–174)

### Step 1 — Get batch size and prepare empty list (lines 124–126)
```python
B = batch["num_values"].shape[0]
tokens = []  # empty basket, will collect 23 tokens one by one
```
The empty list is needed because tokens are built one at a time in a loop.
At the end they're all stacked together.

### Step 2 — Get all positional embeddings at once (line 129)
```python
all_pos_emb = self.pos_embedding(self.positions)  # (23, 64)
```

### Step 3 — Build categorical tokens (lines 132–138)
```python
token = cat_embeddings[feat](local_idx) + all_pos_emb[pos]
# = "what value does SEX have?" + "I am at position 1"
```

### Step 4 — Build PAY tokens (lines 144–154)
```python
state_emb = pay_state_embedding(state_ids)        # "I am delinquent"
sev_emb   = pay_severity_proj(severities)         # "by this much (0.375)"
token     = state_emb + sev_emb + all_pos_emb[pos]  # + "I am at position 4"
```
Three things added together.

### Step 5 — Build numerical tokens (lines 159–169)
```python
feat_emb  = num_feature_embedding(feat_idx)       # "I am LIMIT_BAL"
value_emb = value_proj(values)                    # "my scaled value is 0.83"
token     = value_emb + feat_emb + all_pos_emb[pos]  # + "I am at position 10"
```
Three things added together.

### Step 6 — Prepend [CLS] and stack (lines 171–174)
```python
cls = self.cls_token.expand(B, -1, -1)            # (B, 1, 64)
feature_tokens = torch.stack(tokens, dim=1)       # (B, 23, 64)
return torch.cat([cls, feature_tokens], dim=1)    # (B, 24, 64)
```
[CLS] is glued to the front, giving the final (B, 24, 64) output.

---

## The big picture

```
Raw dataset row (23 numbers)
        ↓
tokenizer.py  →  cat_indices, pay_state_ids, pay_severities, num_values
        ↓
embedding.py  →  (B, 24, 64) tensor
        ↓
self-attention (teammate's job)
        ↓
classification head → P(default)
```

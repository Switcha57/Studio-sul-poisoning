# Quick Reference: Critical Changes for Goodware → Malware Direction

## MOST IMPORTANT CHANGES

### 1. libs/generate_counterfactual.py (Line 23-29)

**BEFORE:**
```python
positive = np.where(ypred == 1)[0]  # Select malware predictions
x_test_true_positive = []
for p in positive:
    xt = x_test[p]
    if y_test[p] == 1:  # Keep only true malware samples
        x_test_true_positive.append(xt)
```

**AFTER:**
```python
negative = np.where(ypred == 0)[0]  # Select goodware predictions  
x_test_true_negative = []  # Or x_test_true_goodware
for p in negative:
    xt = x_test[p]
    if y_test[p] == 0:  # Keep only true goodware samples
        x_test_true_negative.append(xt)
```

### 2. GAMMA.py (Line ~94)

**BEFORE:**
```python
if y_pred[0] == 0:  # Success = malware evaded detection (became goodware)
    res[results[-1]["id"]]["partial"] = True
```

**AFTER:**
```python
if y_pred[0] == 1:  # Success = goodware became malware
    res[results[-1]["id"]]["partial"] = True
```

## Summary
- Change from selecting class 1 (malware) to class 0 (goodware) samples
- Flip success condition from "became goodware" to "became malware"
- Update all variable names containing "positive" to "negative" or "goodware"
- Test with small offset range first (e.g., --offsetmin 100 --offsetmax 105)

The generated counterfactuals will now transform goodware into malware instead of the opposite.

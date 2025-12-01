# Suggestions for Reversing Classification Direction

## Overview
The current OLIVANDER project is designed to take malware samples (class 1) and perturb them to be classified as goodware (class 0). To reverse this direction (goodware → malware), the following modifications are needed:

## Key Files to Modify

### 1. libs/generate_counterfactual.py

**Current Behavior:**
- Selects samples predicted as class 1 (malware)
- Filters for true positives (y_test[p] == 1)
- Generates counterfactuals with `desired_class="opposite"` to flip from malware to goodware

**Required Changes:**
```python
# Line 23: Change from selecting malware (class 1) to goodware (class 0)
# CURRENT:
positive = np.where(ypred == 1)[0]

# CHANGE TO:
negative = np.where(ypred == 0)[0]

# Line 24-29: Change variable names and logic for goodware selection
# CURRENT:
x_test_true_positive = []
for p in positive:
    xt = x_test[p] 
    if y_test[p] == 1:  # Select true malware samples
        x_test_true_positive.append(xt)

# CHANGE TO:
x_test_true_negative = []  # Or rename to x_test_true_goodware
for p in negative:
    xt = x_test[p]
    if y_test[p] == 0:  # Select true goodware samples
        x_test_true_negative.append(xt)

# Update all subsequent references from x_test_true_positive to x_test_true_negative
```

### 2. Main Processing Logic

**File:** main.py (lines 90-130)
**Current Behavior:** Processes malware samples to evade detection
**Required Changes:** Update variable names and comments to reflect goodware→malware direction

### 3. libs/find_adv.py

**Current Behavior:**
- Generates adversarial samples by perturbing malware to evade detection
- Success conditions check for classification as goodware (class 0)

**Required Changes:**
```python
# Line 91: Change condition in the main loop
# CURRENT:
if (not OPTIMIZED and counter == STEP_MATCH) or (counter == 0 and np.argmax(result[0]) == 0):

# CHANGE TO:
if (not OPTIMIZED and counter == STEP_MATCH) or (counter == 0 and np.argmax(result[0]) == 1):

# Line 127: Change success condition
# CURRENT:
if np.argmax(final[0]) == 0:
    print("MODEL EVADED")

# CHANGE TO:
if np.argmax(final[0]) == 1:
    print("MODEL COMPROMISED")  # Or "GOODWARE MADE MALICIOUS"
```

### 4. GAMMA.py

**Current Behavior:**
- Takes malware samples and tries to evade detection (make them appear as goodware)
- Success condition: `if y_pred[0] == 0` (successfully classified as goodware)

**Required Changes:**
```python
# Around line 94: Change success condition
# CURRENT:
if y_pred[0] == 0:  # Successfully evaded (malware → goodware)

# CHANGE TO:
if y_pred[0] == 1:  # Successfully made malicious (goodware → malware)

# Update confidence checks accordingly
# CURRENT:
if confidence[0][0] < confidence[1][0]:  # Check if still detected as malware

# CHANGE TO:
if confidence[0][1] < confidence[1][1]:  # Check if successfully made malicious
```

### 4. Configuration Updates

**File:** config.ini
**Current Setup:** Points to malware samples and goodware for injection
**Required Changes:**
- Consider swapping the roles or updating paths if your dataset structure differs
- Update comments to reflect the new direction

## Detailed Modification Steps

### Step 1: Update generate_counterfactual.py
1. Change `positive = np.where(ypred == 1)[0]` to `negative = np.where(ypred == 0)[0]`
2. Rename `x_test_true_positive` to `x_test_true_negative` or `x_test_true_goodware`
3. Change condition `if y_test[p] == 1:` to `if y_test[p] == 0:`
4. Update all variable references throughout the function
5. Update DataFrame names: `df_test_true_positive` → `df_test_true_negative`

### Step 2: Update GAMMA.py
1. Change success condition from `if y_pred[0] == 0:` to `if y_pred[0] == 1:`
2. Update confidence evaluation logic
3. Update print statements and comments to reflect goodware→malware direction

### Step 3: Update Variable Names and Comments
1. Replace "true_positive" with "true_negative" or "true_goodware" throughout
2. Update comments explaining the process
3. Update output folder names to reflect new direction

### Step 4: Verify Data Flow
1. Ensure the model predictions are interpreted correctly
2. Verify that class 0 = goodware, class 1 = malware consistently
3. Test with a small subset to verify the direction change works

## Testing Strategy

1. **Small Scale Test:** 
   - Run with `--offsetmin 100 --offsetmax 105` to test only 5 samples
   - Verify that goodware samples are being selected and perturbed

2. **Validation:**
   - Check that original samples are classified as class 0 (goodware)
   - Verify that perturbed samples are classified as class 1 (malware)
   - Confirm that the perturbations make sense (goodware becoming malicious)

## Additional Considerations

### Security Implications
- Reversing the direction means creating malware from goodware
- Ensure this is used only for defensive/research purposes
- Consider the ethical implications of generating malicious samples

### Dataset Considerations
- Ensure you have sufficient goodware samples in your test set
- The offset range (offsetmin/offsetmax) should point to goodware samples
- Verify that your training set has balanced classes for counterfactual generation

### Performance Expectations
- Goodware→malware might have different success rates than malware→goodware
- The perturbation patterns might be different
- Consider adjusting hyperparameters (eta, c, etc.) for optimal results

## Files Summary

**Primary files to modify:**
1. `libs/generate_counterfactual.py` - Core logic change (select goodware samples)
2. `libs/find_adv.py` - Success conditions (lines 91 and 127)
3. `GAMMA.py` - Success condition change  
4. `main.py` - Variable name updates
5. `libs/adv_step_mode.py` - Potentially update success evaluation

**Configuration files:**
1. `config.ini` - Update paths if needed

**Testing approach:**
- Start with generate_counterfactual.py changes
- Test counterfactual generation separately
- Then proceed with full pipeline testing

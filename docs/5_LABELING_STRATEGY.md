# 🏷️ 5. Labeling Strategy (The Rules)

> **Target Audience**: Traders, Risk Managers.

QuantGod uses a **Hierarchical Labeling** system based on the "Triple Barrier Method". We do not just predict "Up" or "Down"; we predict **Path-Dependent Outcomes**.

---

## 📐 The Logic (Pseudocode)
For every candle `t`, we look ahead `LABEL_WINDOW_HOURS` (default: 2h).

**Barriers:**
- **Upper Barrier 1**: `Current Close * (1 + 0.8%)`
- **Upper Barrier 2**: `Current Close * (1 + 1.6%)`
- **Lower Barrier**: `Current Close * (1 - 0.75%)`

**Priority Logic (The Hierarchy):**
1.  **STOP (Class 1)**:
    - IF price touches **Lower Barrier** BEFORE touching **Upper Barrier 1**.
    - *Philosophy*: "Safety First". If a trade would hit stop-loss, it is a bad trade, even if it eventually goes up.

2.  **SUPER LONG (Class 3)**:
    - IF price touches **Upper Barrier 1** AND THEN touches **Upper Barrier 2**.
    - *Philosophy*: "Let Winners Run". These are the home runs.

3.  **LONG (Class 2)**:
    - IF price touches **Upper Barrier 1** BUT fails to reach Upper Barrier 2 (or time expires).
    - *Philosophy*: "Base Hits". Valid profit taking.

4.  **NEUTRO (Class 0)**:
    - IF price touches NONE of the barriers within 2 hours.
    - *Philosophy*: "Noise". Do not trade.

---

## 🎯 Target Distribution
The market is mostly noise. A healthy dataset distribution looks like this:
- **NEUTRO**: ~80-90% (Most of the time, do nothing)
- **STOP**: ~5%
- **LONG/SUPER LONG**: ~5-10% (Rare opportunities)

> **Note**: We use `class_weights` in training to handle this imbalance.

---

## 📁 Target Files
Labels are pre-computed and stored to ensure consistency during training loops.
**Path**: `data/processed/targets/target_{month}.csv`

**Columns:**
- `timestamp`: Sync key.
- `close_price`: Reference.
- `label`: `{0, 1, 2, 3}`.

This allows us to iterate on the Model Architecture without re-calculating labels every time.

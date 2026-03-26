# Roadmap: Implementation of AFML Sample Weighting (Chapter 4)

This plan outlines the refactoring required to transition from IID-based training to **Information-Density** based training, as proposed by Marcos Lopez de Prado.

## 🧱 Objective
Correct the "Crowded Bar" problem (overlapping outcomes) by weighting each sample according to its **Uniqueness** and **Absolute Log-Return**.

---

## 📅 Roadmap de Implementação

### Phase A: The Concurrency Matrix (`afml_utils.py`) [NEW]
Create a utility module in `src/cloud/base_model/utils/afml_utils.py` to handle the heavy math of AFML Chapter 4.
- [ ] **Indicator Matrix:** Implement `getIndMatrix` to map which bars influence which labels.
- [ ] **Concurrency Count:** Implement `mpNumCoEvents` to count how many labels are active at time *t*.
- [ ] **Uniqueness ($u_i$):** Implement `mpSampleTW` to calculate the average uniqueness (reciprocal of harmonic average of concurrency).

### Phase B: Labeling Pipeline Upgrade (`run_labelling.py`)
Weight computation should happen at the source to avoid heavy CPU load during training.
- [ ] **Update Parquet Schema:** Add columns `sample_uniqueness` and `sample_weight_return` to the pre-processed parquets.
- [ ] **Return Attribution:** Calculate $w_i$ based on the sum of absolute returns over the event's lifespan $[t_{i,0}, t_{i,1}]$.

### Phase C: Dataset Engine v11.0 (`dataset_utils.py`) [MODIFY]
Update the streaming loader to be weight-aware.
- [ ] **Weight Loading:** Modify `QuantGodLazyDataset` to load the `sample_uniqueness` and `sample_weight_return` columns.
- [ ] **Forwarding:** Ensure `__getitem__` returns `(X, y, weight)`.

### Phase D: Loss Function v2.0 (`losses.py`) [MODIFY]
Adapt the custom loss functions for sample-level priorities.
- [ ] **FocalLoss Extension:** Update `FocalLossWithSmoothing` and `AsymmetricFocalLoss`.
- [ ] **Per-Sample Reduction:** 
    - Change `F.cross_entropy(reduction='none')`.
    - Multiply by `sample_weight`.
    - Apply `focal_modulation`.
    - Return `.mean()`.

### Phase E: Specialist K-Fold Integration (`run_kfold_specialist.py`) [MODIFY]
- [ ] **Training Loop:** Update the training loop to unpack weights from the dataloader.
- [ ] **Optimizer:** Test if higher weights require adaptive gradient scaling for stability.

---

## 📈 Expected Outcome
- **Calibration:** Better alignment between the model's confidence and trade profitability.
- **Robustness:** Reduced overfitting to high-frequency overlapping events (clumping).
- **Audit:** The Auditor (XGBoost) will receive "Pre-Weighted" signals, improving its meta-labeling precision.

**Current Status:** [PLANNING ONLY] - DOCUMENTED ON BRANCH `sample_weighted_classes`

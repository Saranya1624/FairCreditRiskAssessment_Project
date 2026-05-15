# The Fairness Paradox: Fair Credit Risk Assessment for Underserved Populations Using Machine Learning

> **A Dollar-Cost Fairness Framework** — extending ECOA beyond approval-rate parity into financial harm.

**Capstone Project · MS Data Science · Pace University · Seidenberg School of Computer Science and Information Systems · Spring 2026**

**Authors:** Neelima Verma · Saranya Attaluri · Nithish Kumar Vishwanath  
**Faculty Advisor:** Prof. Hetali Chavda

---

## Key Statistics

| Metric | Value |
|--------|-------|
| HMDA NY Applications (2019–2021) | 1.26M |
| ML Models Trained | 10 (3 intervention strategies) |
| Worst Per-Applicant Racial Gap | $6,310 (SVM, 2020) |

---

## Problem

ECOA compliance ≠ financial fairness.

A model can pass every approval-rate test while quietly inflicting greater financial harm on minority borrowers. Under Disparate Impact (DI), a wrongfully denied $100K mortgage and a wrongfully denied $300K mortgage weigh the same — yet the latter represents 3× the economic damage in NPV terms.

> **HMDA 2019 NY:** Non-white avg loan = $191,632 · White avg loan = $136,446 — a **40% gap** that approval-rate fairness ignores.

---

## Dataset

- **Source:** HMDA (Home Mortgage Disclosure Act) New York, 2019–2021
- **Scope:** 1.26M total applications; ~354K+ cleaned records per year
- **Feature Engineering:** 99 → 24 features after SHAP proxy audit
- **Missingness:** 9.1%, median-imputed
- **Observed Gap:** 10.4-point approval rate gap by race, stable across all three years (DI = 0.86 — above ECOA's 0.80 floor)

---

## Proposed Solution: Dollar-Cost Fairness Framework

Convert model errors into **per-applicant dollar harm**, then disaggregate by protected group.

### Error Cost Definitions

**False Negative (Wrongful Denial)**
```
NPV = Σ [ payment_t / (1 + r)^t ] − loan
r = 5% (10-yr Treasury + spread), amortized over term t
```

**False Positive (Missed Default)**
```
Cost = remaining balance × LGD
LGD = 0.40 (Fed residential standard), default at loan_term ÷ 2
```

### Methodology Pipeline

1. **Data + Clean** — HMDA NY, 6-step pipeline
2. **SHAP Proxy Audit** — 5 race proxies removed
3. **Train 10 Models** — LR · SVM · DT · RF · NN
4. **Fairness Evaluation** — DI, EOD, FPR + dollar gap
5. **Cross-Year Validation** — 2019 → 2020 → 2021

---

## Results

### The Fairness Paradox

**SVM achieves the highest Disparate Impact every year — and one of the worst dollar gaps every year.**

DI and dollar-gap show **no significant correlation** across 30 model-year evaluations. Approval-rate fairness and dollar-cost fairness can move in opposite directions simultaneously.

### Model Comparison (2019 HMDA NY — 10-fold stratified CV)

| Model | AUC | DI | EOD | $ Gap | Verdict |
|-------|-----|----|-----|-------|---------|
| LR Baseline | 0.798 | 0.916 | −0.006 | +$3,999 | — |
| LR Balanced | 0.803 | 0.827 | −0.064 | −$927 | — |
| LR Manual Reweighing | 0.802 | 0.898 | −0.010 | +$1,214 | — |
| LR Proxy Removal | 0.787 | 0.940 | +0.007 | +$4,977 | — |
| SVM Linear | 0.791 | **0.943 ★** | +0.011 | +$5,125 | ✗ worst $-gap |
| DT Constrained | 0.993 | 0.868 | −0.006 | −$438 | — |
| Random Forest | 0.993 | 0.880 | +0.002 | **+$52 ★** | ✓ recommended |
| NN + Focal Loss | **0.993** | 0.872 | −0.002 | +$388 | ✓ recommended |

> ★ = best in metric column · ✗ = highest dollar gap · ✓ = recommended for deployment

**Winner: NN + Focal Loss** — Highest AUC (0.993), passes ECOA, lowest relative dollar gap, stable across all three years.

**The Trap:** A regulator looking only at DI would approve SVM — the highest-parity model — while permitting the largest dollar harm.

### Subgroup Analysis

Binary White / Non-White encoding masked a sharper disparity entirely.

| Group | N | FN Rate | Avg Loan | $ / App Gap |
|-------|---|---------|----------|-------------|
| White (baseline) | 53,056 | 3.6% | $136K | — |
| Hispanic | 5,034 | 3.3% | $140K | +$2,903 |
| Black | 5,442 | 4.0% | $172K | +$4,074 |
| Asian | 6,813 | **3.0% (lowest)** | $226K | **+$4,938 ★** |

> Asian applicants have the **lowest denial rate** but bear the **highest financial harm** — a disparity invisible to ECOA's binary compliance check (DI = 0.962, no flag raised).

**Mechanism — Loan-Size Amplification:** Asian-denied loans average $226K, 66% larger than the White average. NPV scales with principal, not denial rate.

---

## Conclusions

1. **Decoupling is structural.** DI and dollar-gap show no significant correlation across 30 model-year evaluations — this is not noise.
2. **Loan-size amplification is the mechanism.** Every approval-rate metric misses principal-driven harm.
3. **Some models clear both bars.** NN + Focal Loss and Random Forest maintain ECOA compliance *and* a relatively low dollar gap — the conventional accuracy/fairness trade-off may not apply to dollar-cost fairness.

**Recommendation:** Adopt dollar-cost metrics as a complementary regulatory standard. ECOA's DI ≥ 0.80 is necessary — but not sufficient — for equitable lending.

---

## Future Work

- Cross-state replication
- Governance & lifecycle monitoring
- Dynamic LGD via LTV
- Causal inference frameworks

---

## Repository Structure

```
├── data/               # HMDA NY data (2019–2021)
├── notebooks/          # EDA, model training, fairness evaluation
├── models/             # Saved model artifacts
├── results/            # Metric tables, fairness plots
└── README.md
```

---

## Citation

If you use this work, please cite:

```
Neelima Verma , Saranya Attaluri , & Nithish Kumar Vishwanath (2026). The Fairness Paradox: Fair Credit Risk
Assessment for Underserved Populations Using Machine Learning. Capstone Project,
MS Data Science, Pace University, Seidenberg School of Computer Science and Information Systems.
Faculty Advisor: Prof. Hetali Chavda.
```

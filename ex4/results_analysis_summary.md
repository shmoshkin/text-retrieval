# Results Analysis Summary: results_1 vs results_2

## Overview

This analysis compares the performance of four ranking models (RM3, Vector, LightGBM, BM25) across two result sets: `results_1` and `results_2`.

## Key Finding

**LightGBM is the only model that differs between the two result sets.** All other models (RM3, Vector, BM25) produce identical results in both folders, suggesting they were unchanged between experiments.

---

## Comprehensive Improvement Comparison: All Methods

### Summary Table: All Models (results_1 → results_2)

| Model | MAP | Rprec | bpref | recip_rank | P@5 | P@10 | P@20 | Status |
|-------|-----|-------|-------|------------|-----|------|------|--------|
| **LightGBM** | +6.7% | +9.4% | +4.2% | +4.3% | +1.1% | +7.4% | +9.5% | ⬆️ **IMPROVED** |
| **RM3** | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | ➡️ **NO CHANGE** |
| **BM25** | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | ➡️ **NO CHANGE** |
| **Vector** | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | ➡️ **NO CHANGE** |

---

## Detailed Model Performance Comparison

### 1. RM3 (run_1_rm3)
**Status: ➡️ NO CHANGE (Identical in both folders)**

| Metric | results_1 | results_2 | Difference | % Change |
|--------|-----------|----------|------------|----------|
| MAP | 0.2699 | 0.2699 | 0.0000 | 0.0% |
| gm_map | 0.1170 | 0.1170 | 0.0000 | 0.0% |
| Rprec | 0.3043 | 0.3043 | 0.0000 | 0.0% |
| bpref | 0.2920 | 0.2920 | 0.0000 | 0.0% |
| recip_rank | 0.6548 | 0.6548 | 0.0000 | 0.0% |
| P@5 | 0.4600 | 0.4600 | 0.0000 | 0.0% |
| P@10 | 0.4060 | 0.4060 | 0.0000 | 0.0% |
| P@15 | 0.3867 | 0.3867 | 0.0000 | 0.0% |
| P@20 | 0.3680 | 0.3680 | 0.0000 | 0.0% |
| P@30 | 0.3287 | 0.3287 | 0.0000 | 0.0% |
| P@100 | 0.2012 | 0.2012 | 0.0000 | 0.0% |

**Conclusion**: No changes between experiments. Model was not modified.

---

### 2. Vector (run_2_vector)
**Status: ➡️ NO CHANGE (Identical in both folders)**

| Metric | results_1 | results_2 | Difference | % Change |
|--------|-----------|----------|------------|----------|
| MAP | 0.1194 | 0.1194 | 0.0000 | 0.0% |
| gm_map | 0.0261 | 0.0261 | 0.0000 | 0.0% |
| Rprec | 0.1464 | 0.1464 | 0.0000 | 0.0% |
| bpref | 0.1514 | 0.1514 | 0.0000 | 0.0% |
| recip_rank | 0.4412 | 0.4412 | 0.0000 | 0.0% |
| P@5 | 0.2640 | 0.2640 | 0.0000 | 0.0% |
| P@10 | 0.2260 | 0.2260 | 0.0000 | 0.0% |
| P@15 | 0.1880 | 0.1880 | 0.0000 | 0.0% |
| P@20 | 0.1690 | 0.1690 | 0.0000 | 0.0% |
| P@30 | 0.1453 | 0.1453 | 0.0000 | 0.0% |
| P@100 | 0.0844 | 0.0844 | 0.0000 | 0.0% |

**Conclusion**: No changes between experiments. This model performs the worst overall and was not modified.

---

### 3. BM25 (run_10_bm25)
**Status: ➡️ NO CHANGE (Identical in both folders)**

| Metric | results_1 | results_2 | Difference | % Change |
|--------|-----------|----------|------------|----------|
| MAP | 0.2454 | 0.2454 | 0.0000 | 0.0% |
| gm_map | 0.1122 | 0.1122 | 0.0000 | 0.0% |
| Rprec | 0.2831 | 0.2831 | 0.0000 | 0.0% |
| bpref | 0.2768 | 0.2768 | 0.0000 | 0.0% |
| recip_rank | 0.6288 | 0.6288 | 0.0000 | 0.0% |
| P@5 | 0.4560 | 0.4560 | 0.0000 | 0.0% |
| P@10 | 0.4140 | 0.4140 | 0.0000 | 0.0% |
| P@15 | 0.3867 | 0.3867 | 0.0000 | 0.0% |
| P@20 | 0.3540 | 0.3540 | 0.0000 | 0.0% |
| P@30 | 0.2993 | 0.2993 | 0.0000 | 0.0% |
| P@100 | 0.1860 | 0.1860 | 0.0000 | 0.0% |

**Conclusion**: No changes between experiments. Model was not modified.

---

### 4. LightGBM (run_3_lightgbm) ⭐
**Status: ⬆️ IMPROVED in results_2**

| Metric | results_1 | results_2 | Difference | % Change |
|--------|-----------|----------|------------|----------|
| MAP | 0.3202 | **0.3418** | **+0.0216** | **+6.7%** |
| gm_map | 0.1785 | **0.2062** | **+0.0277** | **+15.5%** |
| Rprec | 0.3245 | **0.3549** | **+0.0304** | **+9.4%** |
| bpref | 0.3520 | **0.3667** | **+0.0147** | **+4.2%** |
| recip_rank | 0.8590 | **0.8958** | **+0.0368** | **+4.3%** |
| P@5 | 0.7360 | **0.7440** | **+0.0080** | **+1.1%** |
| P@10 | 0.5980 | **0.6420** | **+0.0440** | **+7.4%** |
| P@15 | 0.5120 | **0.5653** | **+0.0533** | **+10.4%** |
| P@20 | 0.4550 | **0.4980** | **+0.0430** | **+9.5%** |
| P@30 | 0.3680 | **0.3853** | **+0.0173** | **+4.7%** |
| P@100 | 0.1718 | **0.1802** | **+0.0084** | **+4.9%** |

**Conclusion**: LightGBM shows consistent improvement across all metrics in `results_2`, with the largest gains in:
- **gm_map**: +15.5% (geometric mean MAP - important for handling query difficulty variance)
- **P@15**: +10.4%
- **P@20**: +9.5%
- **Rprec**: +9.4%

---

## Side-by-Side Metric Comparison: All Methods

### MAP (Mean Average Precision)
| Model | results_1 | results_2 | Change | % Change |
|-------|-----------|----------|--------|----------|
| LightGBM | 0.3202 | **0.3418** | **+0.0216** | **+6.7%** ⬆️ |
| RM3 | 0.2699 | 0.2699 | 0.0000 | 0.0% ➡️ |
| BM25 | 0.2454 | 0.2454 | 0.0000 | 0.0% ➡️ |
| Vector | 0.1194 | 0.1194 | 0.0000 | 0.0% ➡️ |

### Rprec (Precision at R)
| Model | results_1 | results_2 | Change | % Change |
|-------|-----------|----------|--------|----------|
| LightGBM | 0.3245 | **0.3549** | **+0.0304** | **+9.4%** ⬆️ |
| RM3 | 0.3043 | 0.3043 | 0.0000 | 0.0% ➡️ |
| BM25 | 0.2831 | 0.2831 | 0.0000 | 0.0% ➡️ |
| Vector | 0.1464 | 0.1464 | 0.0000 | 0.0% ➡️ |

### bpref (Binary Preference)
| Model | results_1 | results_2 | Change | % Change |
|-------|-----------|----------|--------|----------|
| LightGBM | 0.3520 | **0.3667** | **+0.0147** | **+4.2%** ⬆️ |
| RM3 | 0.2920 | 0.2920 | 0.0000 | 0.0% ➡️ |
| BM25 | 0.2768 | 0.2768 | 0.0000 | 0.0% ➡️ |
| Vector | 0.1514 | 0.1514 | 0.0000 | 0.0% ➡️ |

### Reciprocal Rank
| Model | results_1 | results_2 | Change | % Change |
|-------|-----------|----------|--------|----------|
| LightGBM | 0.8590 | **0.8958** | **+0.0368** | **+4.3%** ⬆️ |
| RM3 | 0.6548 | 0.6548 | 0.0000 | 0.0% ➡️ |
| BM25 | 0.6288 | 0.6288 | 0.0000 | 0.0% ➡️ |
| Vector | 0.4412 | 0.4412 | 0.0000 | 0.0% ➡️ |

### P@5 (Precision at 5)
| Model | results_1 | results_2 | Change | % Change |
|-------|-----------|----------|--------|----------|
| LightGBM | 0.7360 | **0.7440** | **+0.0080** | **+1.1%** ⬆️ |
| RM3 | 0.4600 | 0.4600 | 0.0000 | 0.0% ➡️ |
| BM25 | 0.4560 | 0.4560 | 0.0000 | 0.0% ➡️ |
| Vector | 0.2640 | 0.2640 | 0.0000 | 0.0% ➡️ |

### P@10 (Precision at 10)
| Model | results_1 | results_2 | Change | % Change |
|-------|-----------|----------|--------|----------|
| LightGBM | 0.5980 | **0.6420** | **+0.0440** | **+7.4%** ⬆️ |
| RM3 | 0.4060 | 0.4060 | 0.0000 | 0.0% ➡️ |
| BM25 | 0.4140 | 0.4140 | 0.0000 | 0.0% ➡️ |
| Vector | 0.2260 | 0.2260 | 0.0000 | 0.0% ➡️ |

### P@20 (Precision at 20)
| Model | results_1 | results_2 | Change | % Change |
|-------|-----------|----------|--------|----------|
| LightGBM | 0.4550 | **0.4980** | **+0.0430** | **+9.5%** ⬆️ |
| RM3 | 0.3680 | 0.3680 | 0.0000 | 0.0% ➡️ |
| BM25 | 0.3540 | 0.3540 | 0.0000 | 0.0% ➡️ |
| Vector | 0.1690 | 0.1690 | 0.0000 | 0.0% ➡️ |

### P@100 (Precision at 100)
| Model | results_1 | results_2 | Change | % Change |
|-------|-----------|----------|--------|----------|
| LightGBM | 0.1718 | **0.1802** | **+0.0084** | **+4.9%** ⬆️ |
| RM3 | 0.2012 | 0.2012 | 0.0000 | 0.0% ➡️ |
| BM25 | 0.1860 | 0.1860 | 0.0000 | 0.0% ➡️ |
| Vector | 0.0844 | 0.0844 | 0.0000 | 0.0% ➡️ |

---

## Overall Model Ranking

### results_1 Ranking (by MAP):
1. **LightGBM**: 0.3202
2. **RM3**: 0.2699
3. **BM25**: 0.2454
4. **Vector**: 0.1194

### results_2 Ranking (by MAP):
1. **LightGBM**: 0.3418 ⬆️ (improved)
2. **RM3**: 0.2699
3. **BM25**: 0.2454
4. **Vector**: 0.1194

---

## Detailed Metric Analysis

### Mean Average Precision (MAP)
- **Best**: LightGBM (results_2) = 0.3418
- **Worst**: Vector = 0.1194
- **Gap**: LightGBM is **2.86x better** than Vector

### Precision at Top Positions
- **P@5**: LightGBM (results_2) = 0.7440 (74.4% of top 5 results are relevant)
- **P@10**: LightGBM (results_2) = 0.6420 (64.2% of top 10 results are relevant)
- **P@20**: LightGBM (results_2) = 0.4980 (49.8% of top 20 results are relevant)

### Reciprocal Rank
- **Best**: LightGBM (results_2) = 0.8958 (89.6% of queries have a relevant document in the top position)
- **Worst**: Vector = 0.4412

---

## Insights

1. **LightGBM is the best-performing model** in both result sets, and it improved further in `results_2`.

2. **Vector model performs poorly** - it has the lowest MAP (0.1194) and lowest precision across all cutoffs.

3. **RM3 and BM25 are competitive** - RM3 slightly outperforms BM25, but both are significantly better than Vector.

4. **LightGBM improvement suggests**:
   - Possible hyperparameter tuning
   - Different training data or features
   - Model refinement or retraining
   - Different feature engineering

5. **Consistency of other models** suggests:
   - RM3, Vector, and BM25 were not modified between experiments
   - Only LightGBM was refined/retrained

---

## Recommendations

1. **Use LightGBM from results_2** as the primary ranking model - it shows the best overall performance.

2. **Investigate why Vector performs poorly** - consider feature engineering or different embedding strategies.

3. **Consider ensemble approaches** - combining LightGBM with RM3 or BM25 might yield further improvements.

4. **Analyze query-level performance** - examine which queries benefit most from LightGBM improvements to understand the model's strengths.

---

## Test Statistics

- **Total Queries**: 50
- **Total Relevant Documents**: 4,290
- **Documents Retrieved**: 45,912 (LightGBM, BM25) or 50,000 (RM3, Vector)

---

*Analysis generated from evaluation results in `ex4/results_1/` and `ex4/results_2/`*


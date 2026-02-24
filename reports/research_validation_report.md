# Research Methodology Validation Report

**Purpose**: Verify dataset, tags, judgment criteria, and output artifacts for research reliability.

---

## 1. Dataset & Manifest

- **Total entries**: 29900
- **Goals**: ['goal1', 'goal2', 'normal', 'variant']
- **Scenarios**: ['edge_scorch', 'light_distortion', 'micro_crack']
- **Labels**: [0, 1]
- **Splits**: ['test', 'train', 'val']

### Per-scenario counts

| Scenario | Count |
|----------|-------|
| normal | 21500 |
| crack_in_bending | 2600 |
| light_distortion | 1600 |
| pre_damaged | 1500 |
| thick_panel | 1200 |
| micro_crack | 900 |
| edge_scorch | 600 |

## 2. Train/Val/Test Separation

- Train paths: 20370
- Val paths: 4365
- Test paths: 5165
- **Overlap (train ∩ test)**: 0 ✅

## 3. Judgment Criteria

| Criterion | Value | Source |
|-----------|-------|--------|
| MIN_PRECISION | 0.997 | precision_priority threshold selection |
| Threshold source | Val set | Fallback to test if val too small |
| Ensemble rule | DREAM ∧ PatchCore | Both predict Crack → Crack |
| Normalization fit | Normal-only | Prevents label leakage |

## 4. Output Artifacts

- analysis.json: ✅
- confusion_matrix_dream.png: ✅
- confusion_matrix_patchcore.png: ✅
- confusion_matrix_ensemble.png: ✅
- vector_map_normal.png: ✅
- vector_map_crack.png: ✅
- insights.md: ✅

## 5. Analysis Summary (from analysis.json)

- n_test (rows): 78690
- n_normal: 68625
- n_crack: 10065

- **DREAM**: Precision=0.9967, FP=24, TN=68601
- **PatchCore**: Precision=0.9966, FP=24, TN=68601
- **Ensemble**: Precision=0.9986, FP=10, TN=68615

## 6. Recommendations

1. ~~**edge_scorch**: Add to hard subset for per-scenario evaluation.~~ ✅ Implemented
2. ~~**Vector maps**: Add edge_scorch sample for visualization diversity.~~ ✅ Implemented
3. ~~**Data alignment**: Ensure paper reports match analysis.json dataset scale.~~ ✅ Paper updated
4. ~~**Reference format**: Use consistent citation style (e.g., IEEE or APA).~~ ✅ Implemented

## 7. Iterative Refinement Summary

1. **Methodology validation**: Train/val/test separation verified (no overlap).
2. **Judgment criteria**: Documented and consistent with code.
3. **Paper**: Rewritten for natural flow; references in [1]–[4] format.
4. **Hard subset**: Extended to include edge_scorch; metrics computed per scenario.

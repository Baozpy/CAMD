# CAMD Final Experimental Results

> Status: frozen held-out evaluation. Do not tune prompts, thresholds, retrieval budgets, or scoring rules against these final results.

## 1. Benchmark

- Method-applicable bugs: **98**
- Projects: **Lang, Math, Chart, Time, Mockito**
- Main retrieval budget: **K = 10**
- Multi-agent verification shortlist: **Detector Top-10**

## 2. Program-Wide Retrieval

### Original first-run final retrieval

| Budget | Recall |
|---:|---:|
| 10 | 71/98 (72.45%) |
| 20 | 71/98 (72.45%) |
| 50 | 79/98 (80.61%) |
| 100 | 86/98 (87.76%) |

### Current frozen candidate-pool artifact

| Budget | Recall |
|---:|---:|
| 10 | 71/98 (72.45%) |
| 20 | 71/98 (72.45%) |
| 50 | 80/98 (81.63%) |
| 100 | 86/98 (87.76%) |

**Audit note.** The original unbiased first final retrieval run reported Recall@50 = 79/98. The later frozen candidate-pool export reports 80/98 because Mockito-25 entered the Top-50 pool. The historical run contained 7 failing tests for Mockito-25, while subsequent stable exports contained 6. Because failure evidence contributes to retrieval scoring, the original 79/98 result is retained as the official first-run retrieval metric. Downstream verifier analyses use the internally consistent frozen candidate pools.

## 3. End-to-End Localization

| Method | Top-1 | Top-3 | Top-5 | Top-10 | MRR |
|---|---:|---:|---:|---:|---:|
| Retriever + Detector | 68.37% | 72.45% | 72.45% | 72.45% | 0.7024 |
| Retriever + Detector + Critic + Judge | 67.35% | 72.45% | 72.45% | 72.45% | 0.6990 |

Detector Top-1: **67/98 = 68.37%**.
Judge Top-1: **66/98 = 67.35%**.

## 4. Conditional Localization Quality

Among the **71** bugs whose ground-truth method is present in the Detector Top-10 shortlist:

- Detector Top-1: **67/71 = 94.37%**
- Judge Top-1: **66/71 = 92.96%**
- Detector conditional MRR: **0.9695**
- Judge conditional MRR: **0.9648**

This shows that once the correct method enters the shortlist, the single Detector is already highly effective.

## 5. Final Failure Decomposition

Detector Top-1 failures: **31**

- Retrieval failures: **27 (87.10%)**
- Detector ranking failures after successful retrieval: **4 (12.90%)**

The failure tree is therefore:

```text
98 method-applicable bugs
├── 71 GT retrieved at K=10
│   ├── 67 Detector Top-1 correct
│   └── 4 Detector ranking failures
└── 27 GT missing at K=10
    ├── 9 first recovered at K=50
    ├── 6 first recovered at K=100
    └── 12 still absent at K=100
```

## 6. Candidate-Depth Analysis

Of the 27 K=10 retrieval misses, **15** are recoverable by expanding the frozen candidate pool to K<=100.

Observed recovered GT methods: **18 across 15 recovered bugs**
- Minimum pool/base rank: **23**
- Maximum pool/base rank: **96**
- Mean rank: **56.83**
- Median rank: **42.0**

Admission sources for recovered GT methods:

- base: **18**

All observed recovered GT methods entered through **base retrieval**, indicating that these cases are primarily ranking-depth failures rather than failures of stack/call augmentation.

## 7. Per-Project Failure Decomposition

| Project | Total | Retrieval success | Retrieval miss | Detector Top-1 correct | Detector ranking failure |
|---|---:|---:|---:|---:|---:|
| Chart | 19 | 16 | 3 | 16 | 0 |
| Lang | 19 | 16 | 3 | 16 | 0 |
| Math | 20 | 16 | 4 | 13 | 3 |
| Mockito | 20 | 12 | 8 | 12 | 0 |
| Time | 20 | 11 | 9 | 10 | 1 |

## 8. Multi-Agent Verification Ablation

| Split | Detector Top-1 | Judge Top-1 | Delta |
|---|---:|---:|---:|
| Development | 85.71% | 92.86% | +7.14 pp |
| Held-out final | 68.37% | 67.35% | -1.02 pp |

The Critic/Judge stage improved the development set but did not generalize to the frozen held-out benchmark. It should therefore be reported as an **ablation**, not as the primary CAMD improvement.

## 9. Post-Hoc Expansion-Signal Analysis

| Group | N | Mean Detector p1 | Median p1 | Mean Top1-Top2 margin |
|---|---:|---:|---:|---:|
| retrieved_at_10 | 71 | 0.9441 | 0.9600 | 0.4513 |
| recovered_at_50 | 9 | 0.4189 | 0.3500 | 0.2100 |
| recovered_at_100 | 6 | 0.2100 | 0.2000 | 0.0300 |
| never_recovered | 12 | 0.5308 | 0.4700 | 0.1700 |

**Important methodological warning:** These confidence distributions were inspected after the held-out final results were known. They support hypothesis generation only and must not be used to claim a tuned adaptive-expansion improvement on the same final benchmark.

## 10. Frozen Conclusions

1. **Candidate retrieval is the dominant bottleneck.** 27 of 31 Detector Top-1 failures originate before LLM verification.
2. **The Detector is strong once GT is retrieved.** Conditional Top-1 reaches 94.37%.
3. **Additional multi-agent deliberation does not improve held-out performance.**
4. **Candidate depth matters.** 15 of 27 K=10 misses become reachable in the current frozen candidate pools by K<=100.
5. **The recoverable failures are base-ranking failures.** All observed recovered GT methods were admitted by the base retriever.
6. **Future work should prioritize efficient reranking and candidate coverage**, while separately investigating the 12 cases still absent at K=100.

### Recommended CAMD v1 framing

```text
Whole-program methods
        ↓
Program-wide evidence-aware retrieval
        ↓
Top-K candidate shortlist
        ↓
LLM evidence-grounded Detector
```

Critic/Judge is retained as an experimental verification ablation rather than the main inference path.

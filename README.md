# CAMD: Context-Aware Multi-Agent Software Defect Detection with Large Language Models

CAMD is a program-wide software defect localization framework that combines
evidence-aware candidate retrieval with large language model reasoning.

The project studies a practical question in LLM-based defect localization:

> Given a failing test and an entire buggy program, how can we efficiently identify the method most likely responsible for the failure?

Unlike oracle-style settings that assume the buggy class is already known,
CAMD searches over production methods across the program, constructs a
candidate shortlist using failing-test and structural evidence, and then
applies an LLM-based Detector to perform evidence-grounded defect localization.

The frozen CAMD v1 results show that the dominant bottleneck is
**candidate retrieval**, rather than additional multi-agent deliberation.

---

## Overview

The primary CAMD v1 inference pipeline is:

```text
Whole Program
     |
     v
Failing Test Evidence
     |
     v
Program-Wide Candidate Retrieval
     |
     +---- lexical / test-name evidence
     |
     +---- class / method evidence
     |
     +---- stack-trace evidence
     |
     +---- call-chain augmentation
     |
     v
Top-K Candidate Methods
     |
     v
LLM Detector
     |
     v
Ranked Defect Candidates
```

CAMD also implements an additional multi-agent verification stage:

```text
Detector
   |
   v
Critic
   |
   v
Judge
```

This stage is retained as an experimental ablation.

On the development set, Critic + Judge improved localization performance.
However, the improvement did not transfer to the frozen held-out benchmark.

Therefore, the recommended CAMD v1 inference path is:

```text
Program-Wide Retrieval
        +
LLM Detector
```

---

# Key Results

CAMD is evaluated on five Defects4J projects:

- Apache Commons Lang
- Apache Commons Math
- JFreeChart
- Joda-Time
- Mockito

The frozen benchmark contains:

- 100 selected bugs
- 100 processable bugs
- 98 existing-method localization cases
- 2 method-addition defects

The two method-addition defects are retained in the benchmark artifact but are
excluded from existing-method localization metrics because the corresponding
buggy methods do not exist in the buggy revision.

The main reported localization results therefore use:

```text
N = 98 method-applicable bugs
```

---

## Final End-to-End Localization

| Method | Top-1 | Top-3 | Top-5 | Top-10 | MRR |
|---|---:|---:|---:|---:|---:|
| Retriever + Detector | **68.37%** | **72.45%** | **72.45%** | **72.45%** | **0.7024** |
| Retriever + Detector + Critic + Judge | 67.35% | 72.45% | 72.45% | 72.45% | 0.6990 |

The primary CAMD v1 result is:

```text
Detector Top-1:
67 / 98 = 68.37%
```

---

## Conditional Localization Performance

Among the 71 bugs whose ground-truth method is successfully retrieved into the
Detector Top-10 shortlist:

| Method | Conditional Top-1 | Conditional Top-3 | Conditional MRR |
|---|---:|---:|---:|
| Detector | **94.37%** | **100.00%** | **0.9695** |
| Critic + Judge | 92.96% | 100.00% | 0.9648 |

In other words:

```text
Detector conditional Top-1:
67 / 71 = 94.37%
```

This indicates that once the correct method is available to the LLM,
the Detector is already highly effective.

The main limitation is therefore not primarily LLM verification quality,
but whether the retrieval stage can expose the correct method to the Detector.

---

# Main Finding

The 31 Detector Top-1 failures decompose into:

```text
31 total Detector Top-1 failures
|
+-- 27 retrieval failures
|
+-- 4 Detector ranking failures after successful retrieval
```

Therefore:

```text
Retrieval-related failures: 27 / 31 = 87.10%
Detector-ranking failures:   4 / 31 = 12.90%
```

This leads to the central empirical finding of CAMD v1:

> Candidate retrieval is the dominant bottleneck in program-wide LLM-based defect localization.

---

# Candidate Retrieval

## Program-Wide Search

For each bug, CAMD considers production methods across the program rather than
assuming that the modified class is known in advance.

Let the buggy program contain methods:

$$
\mathcal{M}_b = \{m_1, m_2, \ldots, m_n\}
$$

CAMD assigns each method a retrieval score:

$$
R(m_i, T)
$$

where $T$ denotes failing-test evidence.

The candidate pool is: 
$$ \mathcal{C}_K = \operatorname{TopK}_{m_i \in \mathcal{M}_b} R(m_i, T) $$

The initial retriever combines multiple signals:

$$
R_i =0.35 S_{\text{direct}}+0.20 S_{\text{class}}+0.20 S_{\text{name}}+0.10 S_{\text{test-name}}+0.15 S_{\text{lexical}}
$$

These signals capture:

- direct test references
- class-name similarity
- method-name similarity
- test-name overlap
- lexical similarity

---

## Structural Evidence Augmentation

CAMD additionally augments the base candidate ranking using structural evidence.

The current structural expansion includes:

```text
Failing Test
    |
    v
Stack Evidence
    |
    v
Test Helper Closure
    |
    v
Typed Entry Resolution
    |
    v
Inheritance-Aware Expansion
    |
    v
Virtual Dispatch / Call-Chain Expansion
```

The frozen call-chain configuration uses bounded traversal depths to avoid
uncontrolled candidate explosion.

Structural evidence improves candidate recall in the retrieval development
experiments, particularly at small candidate budgets.

---

# Retrieval Results

## Original First Held-Out Retrieval Run

| Candidate Budget | Recall |
|---:|---:|
| 10 | 71/98 = **72.45%** |
| 20 | 71/98 = **72.45%** |
| 50 | 79/98 = **80.61%** |
| 100 | 86/98 = **87.76%** |

---

## Current Frozen Candidate-Pool Artifact

| Candidate Budget | Recall |
|---:|---:|
| 10 | 71/98 = **72.45%** |
| 20 | 71/98 = **72.45%** |
| 50 | 80/98 = **81.63%** |
| 100 | 86/98 = **87.76%** |

### Audit Note

The original unbiased first final retrieval run reported:

```text
Recall@50 = 79 / 98
```

The later frozen candidate-pool export reports:

```text
Recall@50 = 80 / 98
```

The difference is caused by `Mockito-25`.

The historical first run observed seven failing tests for this bug, while
subsequent stable exports observed six.

Because failing-test evidence participates in retrieval scoring, the candidate
ranking changed and the ground-truth method entered the current Top-50 pool.

To preserve experimental integrity:

- `79/98` is retained as the official first-run `Recall@50`
- downstream verifier analyses use the internally consistent frozen candidate pools
- no result was manually replaced based on favorable performance

---

# Candidate-Depth Analysis

Among the 27 bugs missed at `K=10`:

```text
27 K=10 retrieval misses
|
+-- 9 first recovered at K=50
|
+-- 6 first recovered at K=100
|
+-- 12 still absent at K=100
```

Thus:

```text
15 / 27 = 55.56%
```

of the Top-10 retrieval failures are recoverable in the current frozen
candidate pools by increasing the candidate budget to at most 100 methods.

These 15 cases represent **candidate-ranking depth failures**.

The remaining 12 cases represent harder retrieval failures that require
additional retrieval signals, representations, or structural evidence.

---

## Recovered Ground-Truth Positions

Across the 15 recovered bugs, 18 ground-truth methods were observed in the
larger candidate pools.

Their positions are:

```text
Minimum rank: 23
Maximum rank: 96
Mean rank:    56.83
Median rank:  42
```

All 18 recovered ground-truth methods entered through:

```text
base retrieval
```

rather than stack or call-chain augmentation.

This suggests that these methods were already recognized by the retriever but
were ranked too deeply for the Top-10 shortlist.

---

# LLM Detector

The Detector evaluates each retrieved method independently using:

- method source code
- class and method identity
- source-line information
- failing-test information
- stack-trace evidence
- retrieval metadata
- structural retrieval evidence

Conceptually:

$$
d_i=D(m_i, T, C_i, E_i)
$$

where:

- $m_i$ is the candidate method
- $T$ is failing-test evidence
- $C_i$ is source/context information
- $E_i$ is retrieval and structural evidence

The Detector returns a structured assessment:

```json
{
  "hypothesis": "...",
  "supporting_evidence": [
    "..."
  ],
  "target_defect_probability": 0.0
}
```

Candidates are ranked using the Detector's estimated
`target_defect_probability`.

---

# Multi-Agent Verification Ablation

CAMD also implements:

```text
Detector
   |
   v
Critic
   |
   v
Judge
```

The Critic challenges the Detector's explanation and attempts to distinguish
between:

- a method that is merely on the failing execution path
- an immediate symptom or throw site
- the actual defect responsible for the current failing test

The Critic computes:

$$
c_i=C(m_i, T, d_i)
$$

The Judge then considers both analyses:

$$
s_i=J(T,m_i,d_i,c_i)
$$

where:

- $d_i$ is the Detector assessment
- $c_i$ is the Critic assessment
- $s_i$ is the final Judge score

---

## Development vs Held-Out Result

| Split | Detector Top-1 | Judge Top-1 | Delta |
|---|---:|---:|---:|
| Development | 85.71% | **92.86%** | +7.14 pp |
| Frozen held-out benchmark | **68.37%** | 67.35% | -1.02 pp |

On development data, the multi-agent verifier corrected cases involving
symptom-site confusion.

For example, the Detector could rank an exception-validation method above a
more plausible causal defect location, while the Critic recognized that the
validation method was only the manifestation site.

However, this gain did not generalize to the frozen held-out benchmark.

Final transitions were:

```text
Corrected by Judge: 0
Regressed by Judge: 1
```

Therefore:

> Critic + Judge is treated as an ablation rather than the primary CAMD inference path.

---

# Per-Project Failure Analysis

| Project | Cases | Retrieval Success | Retrieval Miss | Detector Top-1 Correct | Detector Ranking Failure |
|---|---:|---:|---:|---:|---:|
| Chart | 19 | 16 | 3 | 16 | 0 |
| Lang | 19 | 16 | 3 | 16 | 0 |
| Math | 20 | 16 | 4 | 13 | 3 |
| Mockito | 20 | 12 | 8 | 12 | 0 |
| Time | 20 | 11 | 9 | 10 | 1 |

---

## Lang

Once the ground-truth method is retrieved:

```text
Detector Top-1 = 16 / 16
```

All three Top-10 retrieval misses are recoverable by `K=50`.

This suggests that the Lang failures primarily result from candidate-depth
limitations rather than Detector reasoning errors.

---

## Chart

Once retrieved:

```text
Detector Top-1 = 16 / 16
```

The remaining errors are primarily retrieval failures.

---

## Mockito

Once retrieved:

```text
Detector Top-1 = 12 / 12
```

The dominant difficulty is candidate coverage.

At `K=10`, eight Mockito cases fail because no ground-truth method enters the
candidate shortlist.

---

## Time

Time contains the largest number of retrieval failures:

```text
9 / 20
```

Several Time bugs remain absent even from the Top-100 candidate pool.

This makes Time one of the most important projects for future retrieval
analysis.

---

## Math

Math is the main project in which both retrieval and Detector ranking matter:

```text
4 retrieval failures
3 Detector ranking failures
```

It therefore exposes both candidate-coverage and LLM-ranking limitations.

---

# Post-Hoc Expansion Signal Analysis

After the frozen final evaluation, we analyzed whether Detector confidence may
signal that the Top-10 shortlist is incomplete.

This analysis is **post-hoc** and is not used to claim a tuned final
improvement.

| Group | N | Mean Detector Top-1 Probability | Median | Mean Top1-Top2 Margin |
|---|---:|---:|---:|---:|
| Retrieved at K=10 | 71 | 0.9441 | 0.9600 | 0.4513 |
| First recovered at K=50 | 9 | 0.4189 | 0.3500 | 0.2100 |
| First recovered at K=100 | 6 | 0.2100 | 0.2000 | 0.0300 |
| Never recovered by K=100 | 12 | 0.5308 | 0.4700 | 0.1700 |

The recovered-at-100 group shows especially low Detector confidence when the
correct method is absent from the Top-10 shortlist.

However, some unrecovered cases still receive very high Detector confidence.

Therefore:

```text
high Detector confidence
!=
guaranteed retrieval correctness
```

Future adaptive retrieval requires independent validation on a new held-out
benchmark.

---

# Final Failure Tree

The complete CAMD v1 failure decomposition is:

```text
98 method-applicable bugs
|
+-- 71 ground-truth method retrieved at K=10
|   |
|   +-- 67 Detector Top-1 correct
|   |
|   +-- 4 Detector ranking failures
|
+-- 27 ground-truth method missing at K=10
    |
    +-- 9 first recovered at K=50
    |
    +-- 6 additional cases recovered at K=100
    |
    +-- 12 still absent at K=100
```

Equivalently, the 31 end-to-end Detector failures can be decomposed into:

```text
31 failures
|
+-- 15 recoverable candidate-depth failures
|
+-- 12 unresolved retrieval failures
|
+-- 4 Detector ranking failures
```

This decomposition motivates future work on retrieval and efficient reranking.

---

# Frozen CAMD v1 Conclusions

The frozen held-out results support the following conclusions:

1. **Program-wide candidate retrieval is the dominant bottleneck.**

   27 of 31 Detector Top-1 failures occur because the correct method is absent
   from the Top-10 shortlist.

2. **The Detector is strong once the correct candidate is retrieved.**

   Conditional Top-1 reaches:

   ```text
   67 / 71 = 94.37%
   ```

3. **Additional multi-agent deliberation does not improve held-out performance.**

   Critic + Judge improves development performance but slightly reduces
   held-out Top-1.

4. **Candidate depth matters.**

   15 of the 27 Top-10 retrieval misses become reachable in the current frozen
   candidate pools by `K <= 100`.

5. **Recoverable cases are primarily ranking-depth failures.**

   Every observed recovered ground-truth method entered through base retrieval.

6. **Future improvements should prioritize candidate ranking and efficient reranking.**

   Running an expensive LLM over all Top-100 methods is unnecessary if a cheap
   second-stage reranker can compress the candidate set before Detector
   inference.

---

# Recommended Future Direction

A natural CAMD v2 architecture is:

```text
Whole Program
     |
     v
Cheap Base Retrieval
     |
     v
Top-100 Candidate Pool
     |
     v
Cheap Second-Stage Reranker
     |
     v
Top-10 Candidate Pool
     |
     v
LLM Detector
```

A possible adaptive variant is:

```text
Top-10 Retrieval
     |
     v
LLM Detector
     |
     v
Uncertainty / Coverage Gate
     |
     +------ confident ------> return result
     |
     +------ uncertain
               |
               v
       Expand to Top-50 / Top-100
               |
               v
          Cheap Reranker
               |
               v
          LLM Detector
```

The confidence-gating idea is currently only a hypothesis generated from
post-hoc analysis and requires a new held-out evaluation before it can be
reported as an improvement.

---

# Repository Structure

A simplified repository structure is:

```text
CAMD/
├── camd/
│   ├── agents/
│   │   ├── detector_agent.py
│   │   ├── critic_agent.py
│   │   ├── judge_agent.py
│   │   └── models.py
│   │
│   ├── llm/
│   │   └── client.py
│   │
│   ├── retrieval/
│   │   ├── program_method_retriever.py
│   │   └── call_chain_retriever.py
│   │
│   └── verification/
│       ├── frozen_candidate_loader.py
│       ├── detector.py
│       ├── critic.py
│       └── judge.py
│
├── data/
│   └── defects4j/
│       ├── fse_ase_benchmark_v1.json
│       ├── fse_ase_retrieval_dev_v1.json
│       └── checkouts/
│
├── external/
│   └── defects4j/
│
├── results/
│   ├── baseline_predictions.jsonl
│   │
│   ├── defects4j/
│   │   ├── fse_ase_final_retrieval_results.json
│   │   ├── fse_ase_frozen_candidate_pools.json
│   │   └── fse_ase_retrieval_dev_frozen_candidate_pools.json
│   │
│   └── verification/
│       ├── detector/
│       ├── verifier_dev/
│       └── final/
│           ├── detector/
│           ├── verifier/
│           ├── final_verifier_summary.json
│           ├── final_failure_analysis.json
│           ├── candidate_recovery_analysis.json
│           ├── recovered_candidate_positions.json
│           ├── expansion_signal_analysis.json
│           ├── camd_final_experiment_summary.json
│           └── CAMD_FINAL_RESULTS.md
│
├── scripts/
│   ├── run_baseline_batch.py
│   ├── evaluate_candidate_retrieval.py
│   ├── evaluate_final_retrieval.py
│   ├── export_frozen_candidate_pools.py
│   ├── run_frozen_detector.py
│   ├── evaluate_detector_case.py
│   ├── run_detector_dev_batch.py
│   ├── evaluate_detector_dev.py
│   ├── run_verifier_dev_batch.py
│   ├── evaluate_verifier_dev.py
│   ├── analyze_final_failures.py
│   ├── analyze_candidate_recovery.py
│   ├── analyze_recovered_candidate_positions.py
│   ├── analyze_expansion_signals.py
│   └── build_final_experiment_report.py
│
├── requirements.txt
├── .env
└── README.md
```

Local Defects4J checkouts and API credentials should not be committed to the
repository.

---

# Environment

The current experiments were developed with:

```text
Python 3.11
Java 11
Defects4J
macOS / Apple Silicon
```

Create a Python virtual environment:

```bash
python3.11 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# LLM Configuration

Create a local `.env` file:

```text
.env
```

Example:

```bash
OPENAI_API_KEY=your_api_key_here
CAMD_MODEL=gpt-5.5
```

Do **not** commit `.env`.

The LLM client is implemented in:

```text
camd/llm/client.py
```

and uses the configured model for Detector, Critic, and Judge inference.

---

# Java and Defects4J

CAMD uses Defects4J for real-world buggy Java projects.

Java 11 is required by the current local setup.

Example macOS configuration:

```bash
export JAVA_HOME="/opt/homebrew/opt/openjdk@11/libexec/openjdk.jdk/Contents/Home"
export PATH="$JAVA_HOME/bin:$PATH"
```

Configure Defects4J:

```bash
export DEFECTS4J_HOME="$PWD/external/defects4j"
export PATH="$DEFECTS4J_HOME/framework/bin:$PATH"
```

Verify the environment:

```bash
java -version
defects4j info -p Lang
```

---

# Running the Baseline

A small synthetic baseline batch can be executed with:

```bash
PYTHONPATH=. python scripts/run_baseline_batch.py
```

Example cases include:

```text
NullPointerExample.java
SafeStringExample.java
BoundaryBugExample.java
SafeArrayExample.java
```

The baseline results are stored in:

```text
results/baseline_predictions.jsonl
```

---

# Defects4J Benchmark

The frozen final benchmark is stored in:

```text
data/defects4j/fse_ase_benchmark_v1.json
```

The benchmark was sampled once and frozen before final evaluation.

It should **not** be regenerated based on observed experimental performance.

A separate retrieval-development benchmark is stored in:

```text
data/defects4j/fse_ase_retrieval_dev_v1.json
```

The development benchmark is disjoint from:

- the final benchmark
- the earlier Lang 1-20 evaluation set

---

# Export Frozen Candidate Pools

Development candidate pools can be exported with:

```bash
PYTHONPATH=. python scripts/export_frozen_candidate_pools.py \
  --benchmark data/defects4j/fse_ase_retrieval_dev_v1.json \
  --output results/defects4j/fse_ase_retrieval_dev_frozen_candidate_pools.json
```

The frozen final candidate pools are stored in:

```text
results/defects4j/fse_ase_frozen_candidate_pools.json
```

Once exported for final verification, the frozen artifact should be reused
instead of rerunning failing tests or retrieval.

---

# Run Detector Evaluation

## Single Case

Example:

```bash
PYTHONPATH=. python scripts/run_frozen_detector.py \
  --manifest results/defects4j/fse_ase_retrieval_dev_frozen_candidate_pools.json \
  --benchmark-id Lang-21 \
  --budget 10 \
  --include-retrieval-evidence
```

Evaluate the result:

```bash
PYTHONPATH=. python scripts/evaluate_detector_case.py \
  --manifest results/defects4j/fse_ase_retrieval_dev_frozen_candidate_pools.json \
  --benchmark-id Lang-21 \
  --budget 10
```

---

## Full Development Detector Evaluation

Run:

```bash
PYTHONPATH=. python scripts/run_detector_dev_batch.py \
  --budget 10
```

Evaluate:

```bash
PYTHONPATH=. python scripts/evaluate_detector_dev.py \
  --budget 10
```

Frozen development Detector results:

```text
Top-1:  24/28 = 85.71%
Top-3:  26/28 = 92.86%
Top-5:  26/28 = 92.86%
Top-10: 27/28 = 96.43%
MRR:            0.8973
```

Conditional on candidate recall:

```text
Top-1:  24/27 = 88.89%
Top-3:  26/27 = 96.30%
Top-10: 27/27 = 100.00%
Conditional MRR: 0.9306
```

---

# Run Critic + Judge Verification

The development verifier uses:

```text
K_verify = 10
```

This value was frozen using the development set before held-out final
evaluation.

Run:

```bash
PYTHONPATH=. python scripts/run_verifier_dev_batch.py \
  --budget 10 \
  --verify-top-k 10
```

Evaluate:

```bash
PYTHONPATH=. python scripts/evaluate_verifier_dev.py \
  --budget 10 \
  --verify-top-k 10
```

Frozen development results:

```text
Detector Top-1: 24/28 = 85.71%
Judge Top-1:    26/28 = 92.86%

Detector MRR: 0.8973
Judge MRR:    0.9405
```

Top-1 transitions:

```text
Corrected:        2
Regressed:        0
Retained correct: 24
Retained wrong:   2
```

These development results motivated the held-out multi-agent evaluation, but
the improvement did not generalize to the frozen final benchmark.

---

# Frozen Final Evaluation

## Final Detector

```bash
PYTHONPATH=. python scripts/run_detector_dev_batch.py \
  --manifest results/defects4j/fse_ase_frozen_candidate_pools.json \
  --output-dir results/verification/final/detector \
  --budget 10
```

---

## Final Critic + Judge

```bash
PYTHONPATH=. python scripts/run_verifier_dev_batch.py \
  --manifest results/defects4j/fse_ase_frozen_candidate_pools.json \
  --detector-dir results/verification/final/detector \
  --output-dir results/verification/final/verifier \
  --budget 10 \
  --verify-top-k 10
```

---

## Final Evaluation

```bash
PYTHONPATH=. python scripts/evaluate_verifier_dev.py \
  --manifest results/defects4j/fse_ase_frozen_candidate_pools.json \
  --detector-dir results/verification/final/detector \
  --verifier-dir results/verification/final/verifier \
  --budget 10 \
  --verify-top-k 10 \
  --output results/verification/final/final_verifier_summary.json
```

Frozen held-out results:

```text
Retriever Recall@10:
71 / 98 = 72.45%

Detector Top-1:
67 / 98 = 68.37%

Judge Top-1:
66 / 98 = 67.35%

Detector conditional Top-1:
67 / 71 = 94.37%

Judge conditional Top-1:
66 / 71 = 92.96%
```

---

# Offline Failure Analysis

The following analysis scripts do **not** call an LLM.

## Final Failure Decomposition

```bash
PYTHONPATH=. python scripts/analyze_final_failures.py
```

Output:

```text
results/verification/final/final_failure_analysis.json
```

---

## Candidate Recovery Analysis

```bash
PYTHONPATH=. python scripts/analyze_candidate_recovery.py
```

Output:

```text
results/verification/final/candidate_recovery_analysis.json
```

---

## Recovered Candidate Position Analysis

```bash
PYTHONPATH=. python scripts/analyze_recovered_candidate_positions.py
```

Output:

```text
results/verification/final/recovered_candidate_positions.json
```

---

## Expansion-Signal Analysis

```bash
PYTHONPATH=. python scripts/analyze_expansion_signals.py
```

Output:

```text
results/verification/final/expansion_signal_analysis.json
```

---

## Build Frozen Experiment Report

```bash
PYTHONPATH=. python scripts/build_final_experiment_report.py
```

This generates:

```text
results/verification/final/camd_final_experiment_summary.json
results/verification/final/CAMD_FINAL_RESULTS.md
```

---

# Reproducibility Policy

The project follows several rules intended to prevent accidental test-set
tuning.

## Frozen Benchmark

The final benchmark is sampled once and is not regenerated based on observed
performance.

## Frozen Candidate Pools

Final Detector, Critic, and Judge experiments operate on exported frozen
candidate pools.

The retrieval stage and failing tests are not rerun during verifier
evaluation.

## Development / Final Separation

Prompt design and `K_verify` decisions use the development set.

The frozen final benchmark is then used for held-out evaluation.

## No Post-Hoc Final Tuning

Confidence distributions and retrieval-depth patterns discovered after final
evaluation are treated as:

```text
post-hoc failure analysis
```

rather than evidence for a tuned final improvement.

## Cached LLM Outputs

Batch runners save completed results incrementally.

When an experiment is resumed, completed candidate outputs are reused rather
than repeatedly resampled until a favorable result appears.

---

# Experimental Artifacts

Important frozen artifacts include:

```text
results/defects4j/fse_ase_final_retrieval_results.json
results/defects4j/fse_ase_frozen_candidate_pools.json

results/verification/verifier_dev_summary.json

results/verification/final/final_verifier_summary.json
results/verification/final/final_failure_analysis.json
results/verification/final/candidate_recovery_analysis.json
results/verification/final/recovered_candidate_positions.json
results/verification/final/expansion_signal_analysis.json
results/verification/final/camd_final_experiment_summary.json
results/verification/final/CAMD_FINAL_RESULTS.md
```

---

# Known Limitations

## Retrieval Coverage

The main limitation is candidate recall.

At `K=10`, only:

```text
71 / 98 = 72.45%
```

of method-applicable bugs expose at least one ground-truth method to the
Detector.

---

## Expensive Exhaustive LLM Verification

Increasing the candidate budget from 10 to 100 improves retrieval recall, but
running the full LLM Detector over all 100 methods would significantly
increase inference cost.

A cheaper reranking stage is therefore a more promising direction.

---

## High-Confidence Retrieval Failures

Some cases whose ground-truth method is absent from the candidate pool still
receive high Detector confidence on an incorrect method.

Therefore Detector confidence alone is not reliable evidence that candidate
retrieval is complete.

---

## Method-Addition Bugs

CAMD v1 localizes existing executable methods.

Method-addition defects, where the correct method does not exist in the buggy
revision, are currently outside the main localization scope.

---

## Multi-Agent Over-Verification

Critic/Judge may improve individual cases, but the held-out benchmark shows no
consistent overall improvement.

Repeated deliberation over the same evidence can introduce additional ranking
errors.

---

# Research Questions

## RQ1

How effectively can a program-wide retriever reduce an entire Java program to
a small candidate set containing the faulty method?

## RQ2

Once the faulty method is retrieved, how accurately can an LLM Detector rank it
using code, failing-test, and structural evidence?

## RQ3

Does multi-agent verification with Detector, Critic, and Judge improve
localization over a strong single Detector?

## RQ4

Where do remaining failures originate: candidate retrieval, candidate ranking,
or LLM verification?

## RQ5

Can deeper candidate pools be efficiently reranked without applying expensive
LLM reasoning to every method?

---

# Current Research Direction

Based on the frozen CAMD v1 results, the highest-priority next step is not to
add more LLM agents.

Instead, CAMD is moving toward:

```text
efficient candidate reranking
+
adaptive candidate expansion
+
better retrieval coverage
```

A key future objective is to move ground-truth methods currently ranked around:

```text
23 - 96
```

into a much smaller shortlist before expensive LLM inference.

---

# Status

CAMD v1 currently includes:

- [x] Whole-program method extraction
- [x] Program-wide candidate retrieval
- [x] Test / lexical retrieval signals
- [x] Stack-based evidence augmentation
- [x] Call-chain candidate expansion
- [x] Defects4J integration
- [x] Frozen multi-project benchmark
- [x] LLM Detector
- [x] Critic Agent
- [x] Judge Agent
- [x] Development / held-out evaluation
- [x] Candidate-recall analysis
- [x] Failure decomposition
- [x] Candidate-depth analysis
- [x] Post-hoc uncertainty analysis
- [ ] Cheap Top-100 -> Top-10 reranker
- [ ] Independently validated adaptive retrieval
- [ ] Analysis of K=100 unresolved retrieval failures
- [ ] Larger cross-project evaluation

---

# Project Name

**CAMD**

**Context-Aware Multi-Agent Software Defect Detection with Large Language Models**

The project name reflects the broader research system, including the
implemented multi-agent verification components.

However, the frozen CAMD v1 empirical results indicate that the primary
effective inference path is:

```text
Context-Aware Program-Wide Retrieval
                +
        LLM Detector
```

while Critic/Judge serves as an experimental verification ablation.

---

# Citation

A paper citation will be added when the corresponding manuscript is available.

```bibtex
@misc{camd2026,
  title  = {CAMD: Context-Aware Multi-Agent Software Defect Detection with Large Language Models},
  author = {Baozan Yan},
  year   = {2026}
}
```

---

# License

A project license has not yet been finalized.

If this repository is released publicly, an appropriate open-source license
should be added before distribution.

---

# Contact

For questions about the project, experiments, or reproducibility, please open
an issue in this repository.
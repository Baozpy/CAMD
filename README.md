# CAMD: Context-Aware Multi-Agent Software Defect Detection with Large Language Models

CAMD is a research-oriented software defect localization framework that combines **large language models, program context, static analysis, and multi-agent verification** to identify defective methods and localize suspicious code in real-world Java projects.

Rather than asking a single LLM to inspect an entire source file, CAMD decomposes defect localization into multiple stages:

1. candidate method extraction,
2. context-aware candidate ranking,
3. static evidence construction,
4. multi-agent defect verification,
5. line-level localization,
6. adaptive candidate expansion.

The project is evaluated on real bugs from **Defects4J**, with Apache Commons Lang used as the primary experimental target.

In addition to the main CAMD pipeline, the repository contains **local QLoRA fine-tuning baselines using Qwen3.5-9B**, including both binary defect classification and pairwise ranking objectives.

---

## 1. Motivation

Large language models can analyze source code and reason about software defects, but directly applying an LLM to defect localization introduces several problems:

- large source files exceed practical context budgets;
- irrelevant methods introduce substantial noise;
- failing-test information is not always explicitly connected to suspicious methods;
- LLM predictions may be unstable;
- a single model judgment may produce false positives;
- the actual defective method may fall outside a small initial candidate set;
- identifying the correct method does not necessarily identify the exact faulty statement.

CAMD addresses these issues by treating defect localization as a structured pipeline rather than a single LLM call.

The core idea is:

```text
Failing Test
     |
     v
Context Extraction
     |
     v
Candidate Method Generation
     |
     v
Context-Aware Ranking
     |
     v
Static Evidence
     |
     v
Detector
     |
     v
Critic
     |
     v
Judge
     |
     v
Method Ranking
     |
     +----------------------+
     |                      |
     v                      v
Line Localization     Adaptive Expansion
```

---

# 2. Main Research Questions

CAMD investigates several questions.

### RQ1 — Can method-level decomposition improve LLM-based defect localization?

Instead of feeding an entire class to the model, CAMD extracts individual Java methods and ranks them as candidate defect locations.

### RQ2 — Does execution-related context improve candidate ranking?

CAMD incorporates failing-test information and expanded test context when evaluating suspicious methods.

### RQ3 — Can static analysis complement LLM reasoning?

AST-derived evidence is added to candidate representations to expose structural properties such as branches, calls, comparisons, null checks, and exceptions.

### RQ4 — Can multi-agent verification reduce unreliable single-model decisions?

CAMD uses separate Detector, Critic, and Judge roles rather than trusting a single prediction.

### RQ5 — What happens when the true defective method is outside the initial candidate set?

CAMD includes adaptive candidate expansion that increases the search depth when confidence remains low.

### RQ6 — Does lightweight QLoRA fine-tuning improve defect localization?

Binary and pairwise QLoRA baselines are evaluated separately from the main CAMD pipeline.

---

# 3. CAMD Pipeline

## 3.1 Method-Level Decomposition

Java source files are parsed into individual methods.

Each method is represented using information including:

```text
class name
method name
start line
end line
source code
```

This reduces irrelevant context and allows defect localization to operate at method granularity.

---

## 3.2 Failing-Test Context

CAMD extracts failing tests from Defects4J and expands the available test context.

The context may include:

- failing test name,
- failing test method,
- direct helper methods,
- assertion context,
- test-related call structure.

For example:

```text
Failing test
    |
    +-- test method
    |
    +-- directly referenced helper methods
```

This provides a stronger semantic connection between observed failure behavior and candidate production code.

---

## 3.3 Static Analysis Evidence

Each candidate method is analyzed using AST-based static analysis.

Example evidence:

```text
Structural summary:
- Conditional branches
- Loops
- Return statements
- Throw statements

Method calls:
- get
- parse
- compare

Comparisons:
- value == null
- index >= length

Null checks:
- object == null

Thrown exceptions:
- IllegalArgumentException
```

Static evidence is intended to complement semantic LLM reasoning rather than replace it.

---

# 4. Context-Aware Candidate Ranking

Candidate methods are ranked before expensive multi-agent verification.

The ranking stage combines information such as:

- failing-test relevance,
- class and method relationships,
- static evidence,
- candidate structure,
- code context.

This produces a ranked candidate pool:

```text
Candidate 1
Candidate 2
Candidate 3
...
Candidate K
```

Only the highest-ranked candidates are initially sent to the multi-agent verification stage.

---

# 5. Multi-Agent Verification

CAMD uses three reasoning roles.

## Detector

The Detector performs the initial defect analysis.

It evaluates whether the candidate method plausibly explains the current failing test.

```text
Candidate Method
      +
Failing Test Context
      +
Static Evidence
      |
      v
Detector
```

---

## Critic

The Critic independently examines the Detector's reasoning and attempts to identify weaknesses such as:

- unsupported assumptions,
- weak causal connection to the failure,
- alternative candidate explanations,
- false-positive defect claims.

---

## Judge

The Judge receives the available evidence and produces the final candidate assessment.

Conceptually:

```text
Detector Analysis
       +
Critic Analysis
       +
Program Context
       |
       v
     Judge
       |
       v
Final defect probability
```

The resulting Judge probability is used for final candidate ranking.

---

# 6. Adaptive Candidate Expansion

A fixed Top-K candidate pool can fail when the true defective method receives a low initial rank.

CAMD therefore includes an adaptive search mechanism.

Initial evaluation:

```text
Top-5 candidates
```

If the best Judge probability remains below a predefined confidence threshold:

```text
best probability < 0.5
```

CAMD expands the search:

```text
Top-5
  |
  v
Top-10
  |
  v
Top-20
```

Expansion stops when:

- confidence becomes sufficient, or
- the candidate pool is exhausted.

The mechanism is intended as a robustness extension rather than a replacement for the primary CAMD evaluation.

---

# 7. Method-Level Evaluation

The main method-level experiments use Apache Commons Lang bugs from Defects4J.

Lang-2 and Lang-18 are deprecated, leaving:

```text
18 valid bugs
```

The evaluated set is:

```text
Lang:
1, 3, 4, 5, 6, 7, 8, 9, 10,
11, 12, 13, 14, 15, 16, 17,
19, 20
```

Metrics:

- Mean Reciprocal Rank (MRR)
- Top-1
- Top-3
- Top-5
- Top-10

---

# 8. Method-Level Results

## 8.1 Main CAMD Results

| Method | MRR | Top-1 | Top-3 | Top-5 | Top-10 |
|---|---:|---:|---:|---:|---:|
| B1 | 0.6882 | 0.5556 | 0.7222 | 0.8889 | 0.9444 |
| B4 | 0.7435 | 0.6111 | 0.7778 | 0.9444 | 0.9444 |
| **CAMD** | **0.9444** | **0.9444** | **0.9444** | **0.9444** | **0.9444** |

CAMD correctly places a target defective method at rank 1 for:

```text
17 / 18 valid Lang bugs
```

The remaining failure is primarily caused by candidate-generation recall rather than Judge misclassification.

---

# 9. Adaptive Expansion Results

The primary CAMD failure occurs on Lang-20.

The relevant `join` overloads initially appear outside the Top-5 candidate pool.

Observed baseline ranks included:

```text
join(...) -> B4 rank 17
join(...) -> B4 rank 19
```

Adaptive expansion evaluates deeper candidates when Judge confidence remains low.

Final experimental summary:

```text
Evaluated bugs: 18

Triggered:
Lang-6
Lang-20

Candidate pool exhausted:
Lang-6

Recovered Top-1:
Lang-20

Extra candidates evaluated:
15
```

Result:

| Method | MRR | Top-1 | Top-3 | Top-5 | Top-10 | Top-20 |
|---|---:|---:|---:|---:|---:|---:|
| Base CAMD | 0.9444 | 0.9444 | 0.9444 | 0.9444 | 0.9444 | 0.9444 |
| Adaptive CAMD | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |

> **Important:** the adaptive result is a preliminary robustness result on a small set of 18 Lang bugs. The primary CAMD result remains the non-adaptive result of MRR / Top-1 = 0.9444.

---

# 10. Line-Level Localization

After the defective method has been identified, CAMD performs line-level localization.

Three metrics are used.

## Exact Line

The predicted line must exactly match a changed line.

## AST Statement

A prediction is considered correct if it identifies the AST statement containing the changed code.

This accounts for multi-line Java statements.

## ±2 Line Window

A prediction is considered correct if it lies within two lines of the ground-truth change.

---

# 11. Oracle Line-Level Results

Oracle experiments assume the correct defective method is already known.

| Metric | MRR | Top-1 | Top-3 | Top-5 | Top-10 |
|---|---:|---:|---:|---:|---:|
| Exact Line | 0.6667 | 0.6111 | 0.7778 | 0.7778 | 0.7778 |
| AST Statement | **0.8699** | **0.8333** | **0.8889** | **0.9444** | **1.0000** |
| ±2 Lines | 0.8722 | 0.8333 | 0.8889 | 0.9444 | 0.9444 |

The difference between exact-line and AST-statement evaluation demonstrates that source-line matching can underestimate localization quality when a single statement spans multiple physical lines.

---

# 12. End-to-End Line Localization

The complete pipeline includes both method localization and line localization.

| Metric | MRR | Top-1 | Top-3 | Top-5 | Top-10 |
|---|---:|---:|---:|---:|---:|
| Exact Line | 0.6528 | 0.5556 | 0.7222 | 0.7778 | 0.8333 |
| AST Statement | **0.7935** | **0.7222** | **0.8333** | **0.9444** | **0.9444** |
| ±2 Lines | 0.8167 | 0.7778 | 0.8333 | 0.8889 | 0.8889 |

Method-level Top-1:

```text
0.9444
```

Line localization was attempted for:

```text
17 bugs
```

---

# 13. QLoRA Baselines

CAMD also evaluates whether lightweight local fine-tuning can improve software defect localization.

These experiments are intentionally kept **separate from the main CAMD method**.

Base model:

```text
Qwen3.5-9B
```

Training environment:

```text
Apple Silicon
MLX / MLX-LM
4-bit quantized model
LoRA adapters
```

Final QLoRA configuration:

```text
Quantization:       4-bit
LoRA layers:        2
Batch size:         1
Max sequence:       2048
Gradient checkpointing: enabled
Prompt masking:     enabled
Learning rate:      1e-5
Iterations:         20
```

Approximately:

```text
1.328M trainable parameters
~0.015% of model parameters
```

---

# 14. QLoRA Dataset Split

To avoid test leakage:

```text
Training / validation:
Lang 21-65

Final held-out test:
Lang 1-20
```

Lang 1–20 are protected in the dataset-building pipeline and are excluded from training unless explicitly overridden.

The final held-out binary set contains:

```text
18 valid bugs
174 candidate methods
24 positives
150 negatives
```

Deprecated:

```text
Lang-2
Lang-18
```

The held-out test set is not used for:

- training,
- validation,
- checkpoint selection,
- threshold selection,
- prompt tuning,
- sampling-ratio tuning,
- hyperparameter tuning.

---

# 15. Binary QLoRA

The first fine-tuning objective independently classifies each candidate:

```json
{
  "is_target_defect": true
}
```

or:

```json
{
  "is_target_defect": false
}
```

Several class ratios were explored during validation:

```text
1 positive : 8 negatives
1 positive : 4 negatives
1 positive : 1 negative
```

The final binary QLoRA model uses balanced 1:1 hard-negative sampling.

---

# 16. Binary QLoRA Held-Out Results

Held-out Lang 1–20:

```text
18 valid bugs
174 candidates
```

| Metric | Base Qwen | Binary QLoRA |
|---|---:|---:|
| Accuracy | **0.8391** | 0.8333 |
| Precision | **0.4444** | 0.4390 |
| Recall | 0.6667 | **0.7500** |
| F1 | 0.5333 | **0.5538** |
| Specificity | **0.8667** | 0.8467 |
| Balanced Accuracy | 0.7667 | **0.7983** |
| MRR | **0.8370** | 0.8369 |
| Top-1 | 0.7778 | 0.7778 |
| Top-3 | 0.8333 | 0.8333 |
| Top-5 | 0.9444 | 0.9444 |

QLoRA improves positive-defect sensitivity:

```text
Recall:
0.6667 -> 0.7500

F1:
0.5333 -> 0.5538

Balanced Accuracy:
0.7667 -> 0.7983
```

However, method-level ranking remains effectively unchanged.

---

# 17. Pairwise Ranking QLoRA

Because binary classification does not directly optimize ranking, a second fine-tuning task was introduced.

Each example contains two candidate methods from the same bug:

```text
Candidate A
vs
Candidate B
```

with one target defective method and one hard negative.

The model learns:

```json
{
  "preferred_candidate": "A"
}
```

or:

```json
{
  "preferred_candidate": "B"
}
```

A/B ordering is deterministically balanced to prevent positional bias.

Training data:

```text
200 pairs
Preferred A: 100
Preferred B: 100
```

Validation:

```text
52 pairs
Preferred A: 26
Preferred B: 26
```

---

# 18. Pairwise Held-Out Test

Two held-out bugs cannot form positive-negative pairs:

```text
Lang-4
Lang-19
```

because their sampled candidate pools contain only target methods.

Therefore the pairwise held-out evaluation contains:

```text
16 pairable bugs
84 pairs
Preferred A: 42
Preferred B: 42
```

---

# 19. Pairwise QLoRA Held-Out Results

| Metric | Base Qwen | Pairwise QLoRA |
|---|---:|---:|
| Pair Accuracy | 0.8333 | **0.8690** |
| Preferred-A Accuracy | 0.9286 | 0.9286 |
| Preferred-B Accuracy | 0.7381 | **0.8095** |
| Mean Gold Margin | 0.6313 | **0.6661** |
| MRR | 0.8854 | 0.8854 |
| Top-1 | 0.8125 | 0.8125 |
| Top-3 | 1.0000 | 1.0000 |
| Top-5 | 1.0000 | 1.0000 |

Pairwise QLoRA improves local preference discrimination:

```text
Pair Accuracy:
0.8333 -> 0.8690

Preferred-B Accuracy:
0.7381 -> 0.8095

Mean Gold Margin:
0.6313 -> 0.6661
```

However:

```text
MRR:
0.8854 -> 0.8854

Top-1:
0.8125 -> 0.8125
```

The improvement in local pairwise discrimination does not translate into improved global candidate ranking.

---

# 20. QLoRA Findings

The QLoRA experiments reveal an important distinction between **local discrimination** and **global localization**.

Binary QLoRA:

```text
better Recall / F1
        |
        v
no ranking improvement
```

Pairwise QLoRA:

```text
better pair preference accuracy
better confidence margin
        |
        v
no ranking improvement
```

Overall:

> QLoRA changes local defect discrimination and pairwise preference behavior, but these gains do not translate into better method-level localization ranking on the held-out Lang 1–20 benchmark.

This result motivates the context-aware and multi-stage design used by CAMD.

---

# 21. Summary of Main Results

## Method Localization

| Method | MRR | Top-1 | Top-3 | Top-5 |
|---|---:|---:|---:|---:|
| B1 | 0.6882 | 0.5556 | 0.7222 | 0.8889 |
| B4 | 0.7435 | 0.6111 | 0.7778 | 0.9444 |
| **CAMD** | **0.9444** | **0.9444** | **0.9444** | **0.9444** |
| Adaptive CAMD* | **1.0000** | **1.0000** | **1.0000** | **1.0000** |

`*` Preliminary adaptive robustness experiment on 18 Lang bugs.

---

## QLoRA Baselines

| Setting | Base | QLoRA |
|---|---:|---:|
| Binary Recall | 0.6667 | **0.7500** |
| Binary F1 | 0.5333 | **0.5538** |
| Binary MRR | 0.8370 | 0.8369 |
| Binary Top-1 | 0.7778 | 0.7778 |
| Pairwise Accuracy | 0.8333 | **0.8690** |
| Pairwise Gold Margin | 0.6313 | **0.6661** |
| Pairwise MRR | 0.8854 | 0.8854 |
| Pairwise Top-1 | 0.8125 | 0.8125 |

---

# 22. Repository Structure

A simplified project structure is shown below.

```text
CAMD/
│
├── camd/
│   ├── context/
│   │   └── method_extractor.py
│   │
│   ├── evaluation/
│   │   ├── diff_ground_truth.py
│   │   ├── failing_test_extractor.py
│   │   └── test_context_builder.py
│   │
│   ├── static/
│   │   ├── ast_analyzer.py
│   │   └── evidence_builder.py
│   │
│   └── finetuning/
│       ├── dataset_builder.py
│       └── pairwise_dataset_builder.py
│
├── scripts/
│   ├── run_baseline_batch.py
│   │
│   ├── build_qlora_dataset.py
│   ├── prepare_mlx_dataset.py
│   ├── evaluate_qlora.py
│   │
│   ├── build_pairwise_dataset.py
│   ├── prepare_pairwise_mlx_dataset.py
│   ├── evaluate_pairwise_qlora.py
│   ├── evaluate_pairwise_bt.py
│   │
│   ├── build_heldout_qlora_test.py
│   ├── prepare_heldout_mlx_test.py
│   ├── build_heldout_pairwise_test.py
│   ├── prepare_heldout_pairwise_mlx_test.py
│   │
│   └── build_qlora_final_summary.py
│
├── data/
│   ├── defects4j/
│   │   └── checkouts/
│   │
│   └── finetuning/
│       ├── train.jsonl
│       ├── validation.jsonl
│       ├── dataset_manifest.json
│       │
│       ├── pairwise/
│       ├── pairwise_mlx/
│       │
│       └── heldout_lang_1_20/
│           ├── test.jsonl
│           ├── test_manifest.json
│           ├── mlx/
│           ├── pairwise/
│           └── pairwise_mlx/
│
├── results/
│   ├── baseline_predictions.jsonl
│   │
│   ├── defects4j/
│   │   ├── Lang_1_20_final_summary.json
│   │   └── Lang_1_20_adaptive_summary.json
│   │
│   └── qlora/
│       ├── heldout_binary_base_lang1_20.json
│       ├── heldout_binary_qlora11_lang1_20.json
│       ├── heldout_pairwise_base_lang1_20.json
│       ├── heldout_pairwise20_lang1_20.json
│       └── final_summary.json
│
├── external/
│   └── defects4j/
│
├── adapters/
├── models/
├── hf_home/
│
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

Large local assets such as models, Hugging Face caches, Defects4J checkouts, and QLoRA adapters should not be committed to Git.

---

# 23. Environment

The project has been developed and tested on:

```text
macOS
Apple Silicon
Python 3.11
Java 11
Defects4J
MLX / MLX-LM
Qwen3.5-9B
```

---

# 24. Python Setup

Create and activate a virtual environment:

```bash
python3.11 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 25. CAMD Environment Setup

The local environment requires Java 11, Defects4J, Perl dependencies, and the local Hugging Face cache.

Example:

```bash
export JAVA_HOME="/opt/homebrew/opt/openjdk@11/libexec/openjdk.jdk/Contents/Home"

export PERL5LIB="$HOME/perl5/lib/perl5${PERL5LIB:+:$PERL5LIB}"

export DEFECTS4J_HOME="/path/to/CAMD/external/defects4j"

export HF_HOME="/path/to/CAMD/hf_home"

export HF_HUB_OFFLINE=1

export PATH="$JAVA_HOME/bin:$HOME/perl5/bin:$DEFECTS4J_HOME/framework/bin:$PATH"
```

The repository may also use:

```bash
source scripts/setup_env.sh
```

to restore the local environment after opening a new terminal.

Verify:

```bash
java -version
```

```bash
perl -MString::Interpolate -e 'print "Perl OK\n"'
```

```bash
which defects4j
```

---

# 26. Defects4J

Example project information:

```bash
defects4j info -p Lang
```

Checkout a buggy version:

```bash
defects4j checkout \
  -p Lang \
  -v 1b \
  -w data/defects4j/checkouts/Lang_1b
```

---

# 27. Building the QLoRA Dataset

Training and validation use Lang 21–65.

Example:

```bash
PYTHONPATH=. python scripts/build_qlora_dataset.py \
  --project Lang \
  --bug-start 21 \
  --bug-end 65 \
  --max-negatives-per-positive 8
```

The script contains explicit protection for the held-out Lang 1–20 test set.

---

# 28. Preparing MLX Binary Data

```bash
PYTHONPATH=. python scripts/prepare_mlx_dataset.py
```

The conversion applies token-aware context packing with a maximum sequence length of:

```text
2048 tokens
```

---

# 29. Local Qwen Conversion

A Hugging Face Qwen3.5-9B checkpoint can be converted to a local 4-bit MLX model.

Example:

```bash
mlx_lm.convert \
  --hf-path <LOCAL_QWEN_SNAPSHOT> \
  --mlx-path models/qwen35_9b_4bit \
  -q \
  --q-bits 4
```

---

# 30. Binary QLoRA Training

Example final configuration:

```bash
mlx_lm.lora \
  --model models/qwen35_9b_4bit \
  --train \
  --data data/finetuning/mlx \
  --iters 20 \
  --batch-size 1 \
  --num-layers 2 \
  --max-seq-length 2048 \
  --grad-checkpoint \
  --mask-prompt \
  --learning-rate 1e-5 \
  --adapter-path adapters/qwen35_9b_camd_qlora_balanced11_20
```

---

# 31. Pairwise Dataset

Build pairwise examples:

```bash
PYTHONPATH=. python scripts/build_pairwise_dataset.py \
  --negatives-per-positive 4
```

Prepare MLX data:

```bash
PYTHONPATH=. python scripts/prepare_pairwise_mlx_dataset.py
```

---

# 32. Pairwise QLoRA Training

```bash
mlx_lm.lora \
  --model models/qwen35_9b_4bit \
  --train \
  --data data/finetuning/pairwise_mlx \
  --iters 20 \
  --batch-size 1 \
  --num-layers 2 \
  --max-seq-length 2048 \
  --grad-checkpoint \
  --mask-prompt \
  --learning-rate 1e-5 \
  --adapter-path adapters/qwen35_9b_camd_pairwise20
```

---

# 33. Held-Out Binary Evaluation

Build the final Lang 1–20 test set:

```bash
PYTHONPATH=. python scripts/build_heldout_qlora_test.py \
  --max-negatives-per-positive 8
```

Prepare MLX test data:

```bash
PYTHONPATH=. python scripts/prepare_heldout_mlx_test.py
```

Evaluate base Qwen:

```bash
PYTHONPATH=. python scripts/evaluate_qlora.py \
  --no-adapter \
  --metadata data/finetuning/heldout_lang_1_20/test.jsonl \
  --mlx-data data/finetuning/heldout_lang_1_20/mlx/test.jsonl \
  --output results/qlora/heldout_binary_base_lang1_20.json
```

Evaluate binary QLoRA:

```bash
PYTHONPATH=. python scripts/evaluate_qlora.py \
  --adapter adapters/qwen35_9b_camd_qlora_balanced11_20 \
  --metadata data/finetuning/heldout_lang_1_20/test.jsonl \
  --mlx-data data/finetuning/heldout_lang_1_20/mlx/test.jsonl \
  --output results/qlora/heldout_binary_qlora11_lang1_20.json
```

Evaluation compares the conditional log-likelihood of:

```json
{"is_target_defect": true}
```

and:

```json
{"is_target_defect": false}
```

rather than relying on unconstrained text generation.

---

# 34. Held-Out Pairwise Evaluation

Build pairwise held-out data:

```bash
PYTHONPATH=. python scripts/build_heldout_pairwise_test.py
```

Prepare MLX input:

```bash
PYTHONPATH=. python scripts/prepare_heldout_pairwise_mlx_test.py
```

Evaluate pairwise QLoRA:

```bash
PYTHONPATH=. python scripts/evaluate_pairwise_qlora.py \
  --adapter adapters/qwen35_9b_camd_pairwise20 \
  --metadata data/finetuning/heldout_lang_1_20/pairwise/test.jsonl \
  --mlx-validation data/finetuning/heldout_lang_1_20/pairwise_mlx/test.jsonl \
  --output results/qlora/heldout_pairwise20_lang1_20.json
```

Evaluate the base model:

```bash
PYTHONPATH=. python scripts/evaluate_pairwise_qlora.py \
  --no-adapter \
  --metadata data/finetuning/heldout_lang_1_20/pairwise/test.jsonl \
  --mlx-validation data/finetuning/heldout_lang_1_20/pairwise_mlx/test.jsonl \
  --output results/qlora/heldout_pairwise_base_lang1_20.json
```

---

# 35. Building the Final QLoRA Summary

```bash
PYTHONPATH=. python scripts/build_qlora_final_summary.py
```

Output:

```text
results/qlora/final_summary.json
```

This file is the canonical summary of the final held-out QLoRA experiments.

---

# 36. Reproducibility

Important experimental rules:

### No fixed-source leakage

Fixed source code is used only to construct offline ground-truth labels.

Model inference receives:

```text
buggy method
+
failing-test context
+
static evidence
```

The fixed implementation is not included in model prompts.

### Bug-level data splitting

Training and validation are split by bug ID rather than by individual methods.

This prevents methods from the same defect appearing in both train and validation sets.

### Held-out evaluation

Lang 1–20 are reserved for final testing.

Once final test evaluation begins, results are not used to adjust:

- prompts,
- thresholds,
- sampling,
- LoRA layers,
- training iterations,
- ranking aggregation,
- hyperparameters.

---

# 37. Git Ignore Recommendations

Large local files should remain outside version control.

Recommended `.gitignore` entries:

```gitignore
.env
venv/
__pycache__/
*.pyc
.DS_Store

hf_home/
models/
adapters/

data/defects4j/checkouts/
```

Small experimental summaries under:

```text
results/
```

can be committed for reproducibility.

---

# 38. Limitations

Current experiments have several limitations.

## Dataset Scale

The primary formal evaluation currently uses only 18 valid Commons Lang bugs.

Results should therefore not be interpreted as general performance across all Defects4J projects.

## Model Dependence

Multi-agent reasoning still depends on the behavior and calibration of the underlying LLM.

## Candidate Generation

If the defective method does not enter the candidate pool, later reasoning stages cannot recover it without search expansion.

## Line Localization

Exact-line evaluation can be sensitive to formatting and multi-line Java statements.

AST-based statement evaluation partly addresses this issue.

## QLoRA Scale

QLoRA experiments use:

```text
Qwen3.5-9B
2 trainable LoRA layers
20 training iterations
```

They demonstrate that lightweight local fine-tuning can change discrimination behavior, but do not establish that larger-scale fine-tuning cannot improve localization ranking.

## Hardware Constraints

Experiments are designed to run locally on Apple Silicon using unified memory.

This constrains batch size, trainable layers, and sequence length.

---

# 39. Key Findings

The current experiments suggest several conclusions.

### 1. Context-aware decomposition is effective

Method-level decomposition substantially reduces the search space and allows targeted reasoning.

### 2. Static evidence complements semantic reasoning

AST-derived structural evidence provides additional information that is difficult to recover consistently from raw source text alone.

### 3. Multi-agent verification improves localization reliability

Detector, Critic, and Judge roles allow candidate hypotheses to be challenged before final ranking.

### 4. Candidate recall remains important

Lang-20 demonstrates that even a strong verifier cannot identify a method that is excluded from the candidate set.

Adaptive expansion provides one mechanism for handling this failure mode.

### 5. Statement-level evaluation is more informative than strict line matching

Java statements frequently span multiple physical source lines.

### 6. Fine-tuning and localization are not equivalent

QLoRA improves binary defect sensitivity and pairwise preference quality, but these improvements do not automatically produce better global method ranking.

---

# 40. Current Status

Implemented:

```text
✓ LLM-only baseline
✓ Method extraction
✓ Context-aware candidate ranking
✓ Expanded failing-test context
✓ AST-based static evidence
✓ Detector agent
✓ Critic agent
✓ Judge agent
✓ Multi-agent candidate verification
✓ Method-level evaluation
✓ Adaptive candidate expansion
✓ Line-level localization
✓ AST statement evaluation
✓ End-to-end localization
✓ Local Qwen3.5-9B deployment
✓ 4-bit MLX conversion
✓ Binary QLoRA
✓ Pairwise QLoRA
✓ Held-out Lang 1-20 evaluation
✓ Final QLoRA ablation summary
```

---

# 41. Future Work

Possible extensions include:

- evaluation on additional Defects4J projects;
- larger candidate pools;
- dynamic execution traces;
- coverage-based evidence;
- call-graph-aware localization;
- test-to-method dependency modeling;
- learned candidate ranking;
- larger-scale preference optimization;
- multi-project QLoRA training;
- cross-project generalization;
- confidence calibration;
- cost-aware adaptive search;
- stronger line-level localization.

---

# 42. Research Perspective

CAMD is based on the hypothesis that software defect localization is better treated as a **structured reasoning and evidence integration problem** than as a single unconstrained LLM prediction.

The current experimental results support a pipeline in which:

```text
LLM reasoning
+
program structure
+
test context
+
static evidence
+
multi-agent verification
+
adaptive search
```

are combined to improve defect localization reliability.

At the same time, the QLoRA experiments provide an informative negative result: improving local classification or pairwise preference behavior does not necessarily improve global localization ranking.

This distinction is important when designing learning-based defect localization systems.

---

# 43. Disclaimer

CAMD is currently a research prototype.

The reported results are based primarily on Apache Commons Lang bugs from Defects4J and should not be interpreted as production-level defect detection performance or as evidence of generalization to arbitrary software projects.

The adaptive 100% Top-1 result is reported as a preliminary robustness experiment rather than the primary CAMD result.
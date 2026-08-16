from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "qlora"
)

OUTPUT_FILE = (
    RESULTS_DIR
    / "final_summary.json"
)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required result file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def main() -> None:
    # ---------------------------------------------------------
    # Final held-out binary evaluation
    # Lang 1-20, 18 valid bugs
    # ---------------------------------------------------------

    binary_base_path = (
        RESULTS_DIR
        / "heldout_binary_base_lang1_20.json"
    )

    binary_qlora_path = (
        RESULTS_DIR
        / "heldout_binary_qlora11_lang1_20.json"
    )

    binary_base = load_json(
        binary_base_path
    )

    binary_qlora = load_json(
        binary_qlora_path
    )

    binary_base_sample = (
        binary_base["sample_metrics"]
    )

    binary_base_bug = (
        binary_base["bug_metrics"]
    )

    binary_qlora_sample = (
        binary_qlora["sample_metrics"]
    )

    binary_qlora_bug = (
        binary_qlora["bug_metrics"]
    )

    # ---------------------------------------------------------
    # Final held-out pairwise evaluation
    # Lang 1-20, 16 pairable bugs
    # ---------------------------------------------------------

    pairwise_base_path = (
        RESULTS_DIR
        / "heldout_pairwise_base_lang1_20.json"
    )

    pairwise_qlora_path = (
        RESULTS_DIR
        / "heldout_pairwise20_lang1_20.json"
    )

    pairwise_base = load_json(
        pairwise_base_path
    )

    pairwise_qlora = load_json(
        pairwise_qlora_path
    )

    pairwise_base_pair = (
        pairwise_base["pair_metrics"]
    )

    pairwise_base_bug = (
        pairwise_base["bug_metrics"]
    )

    pairwise_qlora_pair = (
        pairwise_qlora["pair_metrics"]
    )

    pairwise_qlora_bug = (
        pairwise_qlora["bug_metrics"]
    )

    # ---------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------

    summary = {
        "experiment": (
            "Qwen3.5-9B QLoRA baselines for "
            "software defect localization"
        ),

        "held_out_policy": {
            "project": "Lang",
            "bug_range": "1-20",
            "deprecated_bugs": [
                2,
                18,
            ],
            "valid_binary_bugs": 18,
            "pairable_bugs": 16,
            "pairwise_excluded_bugs": [
                4,
                19,
            ],
            "note": (
                "Lang 1-20 were reserved for final "
                "evaluation and were not used for "
                "training, validation, checkpoint "
                "selection, prompt tuning, threshold "
                "selection, or hyperparameter tuning."
            ),
        },

        "model": {
            "base_model": (
                "Qwen3.5-9B"
            ),
            "quantization": (
                "4-bit MLX"
            ),
            "qlora": {
                "trainable_layers": 2,
                "batch_size": 1,
                "max_sequence_length": 2048,
                "gradient_checkpointing": True,
                "mask_prompt": True,
                "learning_rate": 1e-5,
                "iterations": 20,
            },
        },

        "binary": {
            "dataset": {
                "held_out_bugs": 18,
                "samples": (
                    binary_base_sample[
                        "samples"
                    ]
                ),
                "threshold": (
                    binary_base_sample[
                        "threshold"
                    ]
                ),
                "training_sampling": (
                    "1 positive : 1 hard negative"
                ),
            },

            "base": {
                "classification": {
                    "accuracy": (
                        binary_base_sample[
                            "accuracy"
                        ]
                    ),
                    "precision": (
                        binary_base_sample[
                            "precision"
                        ]
                    ),
                    "recall": (
                        binary_base_sample[
                            "recall"
                        ]
                    ),
                    "f1": (
                        binary_base_sample[
                            "f1"
                        ]
                    ),
                    "specificity": (
                        binary_base_sample[
                            "specificity"
                        ]
                    ),
                    "balanced_accuracy": (
                        binary_base_sample[
                            "balanced_accuracy"
                        ]
                    ),
                    "tp": (
                        binary_base_sample[
                            "tp"
                        ]
                    ),
                    "tn": (
                        binary_base_sample[
                            "tn"
                        ]
                    ),
                    "fp": (
                        binary_base_sample[
                            "fp"
                        ]
                    ),
                    "fn": (
                        binary_base_sample[
                            "fn"
                        ]
                    ),
                },

                "localization": {
                    "mrr": (
                        binary_base_bug[
                            "mrr"
                        ]
                    ),
                    "top1": (
                        binary_base_bug[
                            "top1"
                        ]
                    ),
                    "top3": (
                        binary_base_bug[
                            "top3"
                        ]
                    ),
                    "top5": (
                        binary_base_bug[
                            "top5"
                        ]
                    ),
                },
            },

            "qlora": {
                "classification": {
                    "accuracy": (
                        binary_qlora_sample[
                            "accuracy"
                        ]
                    ),
                    "precision": (
                        binary_qlora_sample[
                            "precision"
                        ]
                    ),
                    "recall": (
                        binary_qlora_sample[
                            "recall"
                        ]
                    ),
                    "f1": (
                        binary_qlora_sample[
                            "f1"
                        ]
                    ),
                    "specificity": (
                        binary_qlora_sample[
                            "specificity"
                        ]
                    ),
                    "balanced_accuracy": (
                        binary_qlora_sample[
                            "balanced_accuracy"
                        ]
                    ),
                    "tp": (
                        binary_qlora_sample[
                            "tp"
                        ]
                    ),
                    "tn": (
                        binary_qlora_sample[
                            "tn"
                        ]
                    ),
                    "fp": (
                        binary_qlora_sample[
                            "fp"
                        ]
                    ),
                    "fn": (
                        binary_qlora_sample[
                            "fn"
                        ]
                    ),
                },

                "localization": {
                    "mrr": (
                        binary_qlora_bug[
                            "mrr"
                        ]
                    ),
                    "top1": (
                        binary_qlora_bug[
                            "top1"
                        ]
                    ),
                    "top3": (
                        binary_qlora_bug[
                            "top3"
                        ]
                    ),
                    "top5": (
                        binary_qlora_bug[
                            "top5"
                        ]
                    ),
                },
            },

            "delta": {
                "recall": (
                    binary_qlora_sample[
                        "recall"
                    ]
                    - binary_base_sample[
                        "recall"
                    ]
                ),
                "f1": (
                    binary_qlora_sample[
                        "f1"
                    ]
                    - binary_base_sample[
                        "f1"
                    ]
                ),
                "balanced_accuracy": (
                    binary_qlora_sample[
                        "balanced_accuracy"
                    ]
                    - binary_base_sample[
                        "balanced_accuracy"
                    ]
                ),
                "mrr": (
                    binary_qlora_bug[
                        "mrr"
                    ]
                    - binary_base_bug[
                        "mrr"
                    ]
                ),
                "top1": (
                    binary_qlora_bug[
                        "top1"
                    ]
                    - binary_base_bug[
                        "top1"
                    ]
                ),
            },
        },

        "pairwise": {
            "dataset": {
                "held_out_pairable_bugs": (
                    pairwise_base_bug[
                        "bugs"
                    ]
                ),
                "pairs": (
                    pairwise_base_pair[
                        "pairs"
                    ]
                ),
                "negatives_per_positive": 4,
                "aggregation": (
                    "mean expected win probability"
                ),
            },

            "base": {
                "preference": {
                    "accuracy": (
                        pairwise_base_pair[
                            "accuracy"
                        ]
                    ),
                    "preferred_a_accuracy": (
                        pairwise_base_pair[
                            "preferred_a_accuracy"
                        ]
                    ),
                    "preferred_b_accuracy": (
                        pairwise_base_pair[
                            "preferred_b_accuracy"
                        ]
                    ),
                    "mean_gold_margin": (
                        pairwise_base_pair[
                            "mean_gold_margin"
                        ]
                    ),
                },

                "localization": {
                    "mrr": (
                        pairwise_base_bug[
                            "mrr"
                        ]
                    ),
                    "top1": (
                        pairwise_base_bug[
                            "top1"
                        ]
                    ),
                    "top3": (
                        pairwise_base_bug[
                            "top3"
                        ]
                    ),
                    "top5": (
                        pairwise_base_bug[
                            "top5"
                        ]
                    ),
                },
            },

            "qlora": {
                "preference": {
                    "accuracy": (
                        pairwise_qlora_pair[
                            "accuracy"
                        ]
                    ),
                    "preferred_a_accuracy": (
                        pairwise_qlora_pair[
                            "preferred_a_accuracy"
                        ]
                    ),
                    "preferred_b_accuracy": (
                        pairwise_qlora_pair[
                            "preferred_b_accuracy"
                        ]
                    ),
                    "mean_gold_margin": (
                        pairwise_qlora_pair[
                            "mean_gold_margin"
                        ]
                    ),
                },

                "localization": {
                    "mrr": (
                        pairwise_qlora_bug[
                            "mrr"
                        ]
                    ),
                    "top1": (
                        pairwise_qlora_bug[
                            "top1"
                        ]
                    ),
                    "top3": (
                        pairwise_qlora_bug[
                            "top3"
                        ]
                    ),
                    "top5": (
                        pairwise_qlora_bug[
                            "top5"
                        ]
                    ),
                },
            },

            "delta": {
                "pair_accuracy": (
                    pairwise_qlora_pair[
                        "accuracy"
                    ]
                    - pairwise_base_pair[
                        "accuracy"
                    ]
                ),
                "preferred_b_accuracy": (
                    pairwise_qlora_pair[
                        "preferred_b_accuracy"
                    ]
                    - pairwise_base_pair[
                        "preferred_b_accuracy"
                    ]
                ),
                "mean_gold_margin": (
                    pairwise_qlora_pair[
                        "mean_gold_margin"
                    ]
                    - pairwise_base_pair[
                        "mean_gold_margin"
                    ]
                ),
                "mrr": (
                    pairwise_qlora_bug[
                        "mrr"
                    ]
                    - pairwise_base_bug[
                        "mrr"
                    ]
                ),
                "top1": (
                    pairwise_qlora_bug[
                        "top1"
                    ]
                    - pairwise_base_bug[
                        "top1"
                    ]
                ),
            },
        },

        "conclusion": {
            "binary": (
                "Class-balanced binary QLoRA improved "
                "defect sensitivity and F1 but did not "
                "improve method-level localization ranking."
            ),

            "pairwise": (
                "Pairwise QLoRA improved pairwise "
                "preference accuracy and confidence margin "
                "but did not improve method-level "
                "localization ranking."
            ),

            "overall": (
                "QLoRA changed local discrimination and "
                "preference behavior, but the gains did not "
                "translate into better global method ranking "
                "on the held-out Lang 1-20 benchmark."
            ),
        },
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 100)
    print("QLoRA Final Summary")
    print("=" * 100)

    print()
    print("Binary held-out")
    print(
        "  Base F1: "
        f"{binary_base_sample['f1']:.4f}"
    )
    print(
        "  QLoRA F1: "
        f"{binary_qlora_sample['f1']:.4f}"
    )
    print(
        "  Base Recall: "
        f"{binary_base_sample['recall']:.4f}"
    )
    print(
        "  QLoRA Recall: "
        f"{binary_qlora_sample['recall']:.4f}"
    )
    print(
        "  Base MRR: "
        f"{binary_base_bug['mrr']:.4f}"
    )
    print(
        "  QLoRA MRR: "
        f"{binary_qlora_bug['mrr']:.4f}"
    )
    print(
        "  Base Top-1: "
        f"{binary_base_bug['top1']:.4f}"
    )
    print(
        "  QLoRA Top-1: "
        f"{binary_qlora_bug['top1']:.4f}"
    )

    print()
    print("Pairwise held-out")
    print(
        "  Base Pair Accuracy: "
        f"{pairwise_base_pair['accuracy']:.4f}"
    )
    print(
        "  QLoRA Pair Accuracy: "
        f"{pairwise_qlora_pair['accuracy']:.4f}"
    )
    print(
        "  Base Gold Margin: "
        f"{pairwise_base_pair['mean_gold_margin']:.4f}"
    )
    print(
        "  QLoRA Gold Margin: "
        f"{pairwise_qlora_pair['mean_gold_margin']:.4f}"
    )
    print(
        "  Base MRR: "
        f"{pairwise_base_bug['mrr']:.4f}"
    )
    print(
        "  QLoRA MRR: "
        f"{pairwise_qlora_bug['mrr']:.4f}"
    )
    print(
        "  Base Top-1: "
        f"{pairwise_base_bug['top1']:.4f}"
    )
    print(
        "  QLoRA Top-1: "
        f"{pairwise_qlora_bug['top1']:.4f}"
    )

    print()
    print("Saved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
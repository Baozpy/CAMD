from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)

    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            rows.append(json.loads(line))

    return rows


def save_jsonl(
    rows: list[dict[str, Any]],
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def extract_section(
    text: str,
    start_marker: str,
    end_marker: str | None = None,
) -> str:
    start_index = text.find(start_marker)

    if start_index == -1:
        return ""

    start_index += len(start_marker)

    if end_marker is None:
        return text[start_index:].strip()

    end_index = text.find(
        end_marker,
        start_index,
    )

    if end_index == -1:
        return text[start_index:].strip()

    return text[start_index:end_index].strip()


def extract_candidate_source(sample: dict[str, Any]) -> str:
    text = sample["input"]

    candidate_section = extract_section(
        text,
        "CANDIDATE METHOD\n================",
        "STATIC EVIDENCE",
    )

    source_marker = "Source:\n"
    source_index = candidate_section.find(source_marker)

    if source_index == -1:
        return candidate_section.strip()

    return candidate_section[
        source_index + len(source_marker):
    ].strip()


def extract_static_evidence(sample: dict[str, Any]) -> str:
    text = sample["input"]

    section = extract_section(
        text,
        "STATIC EVIDENCE\n===============",
        "FAILING TEST CONTEXT",
    )

    return section.strip()


def extract_failing_test_context(sample: dict[str, Any]) -> str:
    text = sample["input"]

    section = extract_section(
        text,
        "FAILING TEST CONTEXT\n====================",
        "Return whether this candidate method",
    )

    return section.strip()


def candidate_identity(sample: dict[str, Any]) -> str:
    return (
        f"{sample['class_name']}::"
        f"{sample['method_name']}@"
        f"{sample['start_line']}-{sample['end_line']}"
    )


def build_pair_prompt(
    positive: dict[str, Any],
    negative: dict[str, Any],
    preferred_candidate: str,
) -> dict[str, Any]:
    if preferred_candidate == "A":
        candidate_a = positive
        candidate_b = negative
    else:
        candidate_a = negative
        candidate_b = positive

    failing_test_context = extract_failing_test_context(
        positive
    )

    candidate_a_method = extract_candidate_source(
        candidate_a
    )
    candidate_b_method = extract_candidate_source(
        candidate_b
    )

    candidate_a_static = extract_static_evidence(
        candidate_a
    )
    candidate_b_static = extract_static_evidence(
        candidate_b
    )

    prompt = f"""You are given two candidate Java methods from the same buggy program.

Your task is to determine which candidate is MORE LIKELY to contain the target defect responsible for the CURRENT failing test.

Compare Candidate A and Candidate B directly.

Do not search for unrelated defects.
Do not propose a patch.
Choose exactly one candidate.

PROJECT
=======

{positive['project']}-{positive['bug_id']}


FAILING TEST CONTEXT
====================

{failing_test_context}


CANDIDATE A
===========

Class:
{candidate_a['class_name']}

Method:
{candidate_a['method_name']}

Lines:
{candidate_a['start_line']}-{candidate_a['end_line']}

{candidate_a_method}


CANDIDATE A STATIC EVIDENCE
===========================

{candidate_a_static}


CANDIDATE B
===========

Class:
{candidate_b['class_name']}

Method:
{candidate_b['method_name']}

Lines:
{candidate_b['start_line']}-{candidate_b['end_line']}

{candidate_b_method}


CANDIDATE B STATIC EVIDENCE
===========================

{candidate_b_static}


Return which candidate is more likely to contain the target defect for the current failing test."""

    return {
        "project": positive["project"],
        "bug_id": positive["bug_id"],
        "candidate_a": {
            "class_name": candidate_a["class_name"],
            "method_name": candidate_a["method_name"],
            "start_line": candidate_a["start_line"],
            "end_line": candidate_a["end_line"],
            "is_target_defect": bool(
                candidate_a["is_target_defect"]
            ),
            "identity": candidate_identity(
                candidate_a
            ),
        },
        "candidate_b": {
            "class_name": candidate_b["class_name"],
            "method_name": candidate_b["method_name"],
            "start_line": candidate_b["start_line"],
            "end_line": candidate_b["end_line"],
            "is_target_defect": bool(
                candidate_b["is_target_defect"]
            ),
            "identity": candidate_identity(
                candidate_b
            ),
        },
        "preferred_candidate": preferred_candidate,
        "input": prompt,
        "output": {
            "preferred_candidate": preferred_candidate,
        },
    }


def build_pairwise_dataset(
    samples: list[dict[str, Any]],
    *,
    negatives_per_positive: int = 4,
    seed: int = 42,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)

    by_bug: dict[
        tuple[str, int],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for sample in samples:
        key = (
            sample["project"],
            int(sample["bug_id"]),
        )

        by_bug[key].append(sample)

    pairs: list[dict[str, Any]] = []

    for key in sorted(by_bug):
        bug_samples = by_bug[key]

        positives = [
            sample
            for sample in bug_samples
            if sample["is_target_defect"]
        ]

        negatives = [
            sample
            for sample in bug_samples
            if not sample["is_target_defect"]
        ]

        if not positives or not negatives:
            continue

        for positive_index, positive in enumerate(
            positives
        ):
            ranked_negatives = sorted(
                negatives,
                key=lambda negative: (
                    abs(
                        int(
                            negative.get(
                                "method_length",
                                0,
                            )
                        )
                        - int(
                            positive.get(
                                "method_length",
                                0,
                            )
                        )
                    ),
                    negative["class_name"]
                    != positive["class_name"],
                    negative["method_name"]
                    != positive["method_name"],
                    negative["start_line"],
                ),
            )

            selected_negatives = (
                ranked_negatives[
                    :negatives_per_positive
                ]
            )

            for negative_index, negative in enumerate(
                selected_negatives
            ):
                swap = (
                    positive_index
                    + negative_index
                    + int(positive["bug_id"])
                    + seed
                ) % 2

                preferred_candidate = (
                    "A"
                    if swap == 0
                    else "B"
                )

                pair = build_pair_prompt(
                    positive,
                    negative,
                    preferred_candidate,
                )

                pairs.append(pair)

    rng.shuffle(pairs)

    return pairs


def summarize_pairs(
    pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    bugs = sorted(
        {
            (
                pair["project"],
                int(pair["bug_id"]),
            )
            for pair in pairs
        }
    )

    preferred_a = sum(
        pair["preferred_candidate"] == "A"
        for pair in pairs
    )

    preferred_b = len(pairs) - preferred_a

    return {
        "bugs": len(bugs),
        "pairs": len(pairs),
        "preferred_a": preferred_a,
        "preferred_b": preferred_b,
        "preferred_a_ratio": (
            preferred_a / len(pairs)
            if pairs
            else 0.0
        ),
        "bug_ids": [
            bug_id
            for _, bug_id in bugs
        ],
    }
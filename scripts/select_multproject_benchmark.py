from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "defects4j"
    / "multi_project_manifest.json"
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "defects4j"
    / "fse_ase_benchmark_v1.json"
)

DEFAULT_SEED = 2026

DEFAULT_PER_PROJECT = 20

PROJECT_ORDER = [
    "Lang",
    "Math",
    "Chart",
    "Time",
    "Mockito",
]

# -------------------------------------------------------------
# Lang 1-20 have already been used during CAMD development
# and detailed failure analysis.
#
# Active bugs among 1-20:
# 1,3,...,17,19,20
#
# We exclude the entire numerical range 1-20 rather than only
# the active IDs to make the policy simple and explicit.
# -------------------------------------------------------------

DEVELOPMENT_EXCLUSIONS: dict[str, set[int]] = {
    "Lang": set(range(1, 21)),
    "Math": set(),
    "Chart": set(),
    "Time": set(),
    "Mockito": set(),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select a deterministic, stratified "
            "multi-project benchmark for CAMD."
        )
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--per-project",
        type=int,
        default=DEFAULT_PER_PROJECT,
    )

    return parser.parse_args()


def load_json(
    path: Path,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def eligible_bug_ids(
    project: str,
    active_bug_ids: list[int],
) -> list[int]:
    excluded = DEVELOPMENT_EXCLUSIONS.get(
        project,
        set(),
    )

    return [
        bug_id
        for bug_id in active_bug_ids
        if bug_id not in excluded
    ]


def deterministic_sample(
    *,
    project: str,
    bug_ids: list[int],
    count: int,
    global_seed: int,
) -> list[int]:
    if len(bug_ids) < count:
        raise ValueError(
            f"{project} has only {len(bug_ids)} "
            f"eligible bugs, but {count} were requested."
        )

    # Project-specific seed prevents the sample for one project
    # from changing if another project is added or removed later.
    project_seed = (
        global_seed
        + sum(
            (index + 1) * ord(char)
            for index, char in enumerate(project)
        )
    )

    rng = random.Random(
        project_seed
    )

    selected = rng.sample(
        bug_ids,
        count,
    )

    return sorted(
        selected
    )


def main() -> None:
    args = parse_args()

    manifest = load_json(
        args.manifest
    )

    manifest_projects = manifest[
        "projects"
    ]

    print()
    print("=" * 100)
    print(
        "CAMD FSE/ASE Multi-Project "
        "Benchmark Selection"
    )
    print("=" * 100)

    benchmark_projects: dict[
        str,
        dict[str, Any]
    ] = {}

    all_entries: list[
        dict[str, Any]
    ] = []

    total_eligible = 0
    total_selected = 0

    for project in PROJECT_ORDER:
        if project not in manifest_projects:
            raise KeyError(
                f"{project} not found in manifest."
            )

        source_entry = manifest_projects[
            project
        ]

        active_ids = [
            int(x)
            for x in source_entry[
                "active_bug_ids"
            ]
        ]

        eligible_ids = eligible_bug_ids(
            project,
            active_ids,
        )

        selected_ids = deterministic_sample(
            project=project,
            bug_ids=eligible_ids,
            count=args.per_project,
            global_seed=args.seed,
        )

        exclusions = sorted(
            set(active_ids)
            & DEVELOPMENT_EXCLUSIONS.get(
                project,
                set(),
            )
        )

        benchmark_projects[
            project
        ] = {
            "active_bug_count": len(
                active_ids
            ),
            "eligible_bug_count": len(
                eligible_ids
            ),
            "development_excluded_bug_ids": (
                exclusions
            ),
            "selected_bug_count": len(
                selected_ids
            ),
            "selected_bug_ids": (
                selected_ids
            ),
        }

        for bug_id in selected_ids:
            all_entries.append(
                {
                    "project": project,
                    "bug_id": bug_id,
                    "version": f"{bug_id}b",
                    "benchmark_id": (
                        f"{project}-{bug_id}"
                    ),
                }
            )

        total_eligible += len(
            eligible_ids
        )

        total_selected += len(
            selected_ids
        )

        print()
        print(
            f"[{project}]"
        )

        print(
            "  Active: "
            f"{len(active_ids)}"
        )

        print(
            "  Development excluded: "
            f"{len(exclusions)}"
        )

        if exclusions:
            print(
                "  Excluded IDs: "
                f"{exclusions}"
            )

        print(
            "  Eligible: "
            f"{len(eligible_ids)}"
        )

        print(
            "  Selected: "
            f"{len(selected_ids)}"
        )

        print(
            "  IDs: "
            f"{selected_ids}"
        )

    benchmark = {
        "name": (
            "CAMD FSE/ASE "
            "Multi-Project Benchmark v1"
        ),

        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "selection_policy": {
            "strategy": (
                "deterministic stratified random sampling"
            ),

            "seed": (
                args.seed
            ),

            "bugs_per_project": (
                args.per_project
            ),

            "projects": (
                PROJECT_ORDER
            ),

            "development_policy": (
                "Lang bug IDs 1-20 are excluded "
                "from the multi-project pilot because "
                "they were previously used during CAMD "
                "development and failure analysis."
            ),

            "selection_freeze_policy": (
                "After this benchmark is generated, "
                "the selected bug IDs must not be changed "
                "based on CAMD performance."
            ),
        },

        "projects": (
            benchmark_projects
        ),

        "entries": (
            all_entries
        ),

        "summary": {
            "project_count": len(
                benchmark_projects
            ),

            "total_eligible_bugs": (
                total_eligible
            ),

            "total_selected_bugs": (
                total_selected
            ),
        },
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            benchmark,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 100)
    print("Benchmark Summary")
    print("=" * 100)

    print(
        "Projects: "
        f"{len(benchmark_projects)}"
    )

    print(
        "Total eligible bugs: "
        f"{total_eligible}"
    )

    print(
        "Selected bugs: "
        f"{total_selected}"
    )

    print(
        "Seed: "
        f"{args.seed}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "  Once CAMD evaluation begins, "
        "do not regenerate this benchmark "
        "with another seed based on results."
    )

    print()
    print(
        "Saved:"
    )

    print(
        args.output
    )


if __name__ == "__main__":
    main()
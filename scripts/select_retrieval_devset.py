from __future__ import annotations

import json
import random
from pathlib import Path


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

MANIFEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "defects4j"
    / "multi_project_manifest.json"
)

FINAL_BENCHMARK_FILE = (
    PROJECT_ROOT
    / "data"
    / "defects4j"
    / "fse_ase_benchmark_v1.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "defects4j"
    / "fse_ase_retrieval_dev_v1.json"
)

PROJECTS = [
    "Lang",
    "Math",
    "Chart",
    "Time",
    "Mockito",
]

SEED = 20260816
PER_PROJECT = 6


def load_json(
    path: Path,
) -> dict:

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


def save_json(
    path: Path,
    data: dict,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )


def main() -> None:

    manifest = load_json(
        MANIFEST_FILE
    )

    final_benchmark = load_json(
        FINAL_BENCHMARK_FILE
    )

    final_ids = {
        (
            item["project"],
            int(item["bug_id"]),
        )
        for item in final_benchmark[
            "entries"
        ]
    }

    # Lang 1-20 were previously used for
    # CAMD development/failure analysis.
    prior_lang_dev = {
        ("Lang", bug_id)
        for bug_id in range(
            1,
            21,
        )
    }

    excluded_ids = (
        final_ids
        | prior_lang_dev
    )

    rng = random.Random(
        SEED
    )

    selected_entries = []

    project_summary = {}

    for project in PROJECTS:

        project_record = (
            manifest[
                "projects"
            ][
                project
            ]
        )

        active_bugs = [
            int(bug_id)
            for bug_id in (
                project_record[
                    "active_bug_ids"
                ]
            )
        ]

        eligible = [
            bug_id
            for bug_id in active_bugs
            if (
                project,
                bug_id,
            )
            not in excluded_ids
        ]

        if len(
            eligible
        ) < PER_PROJECT:

            raise RuntimeError(
                f"{project} has only "
                f"{len(eligible)} eligible "
                "development bugs."
            )

        selected = sorted(
            rng.sample(
                eligible,
                PER_PROJECT,
            )
        )

        project_summary[
            project
        ] = {
            "active": (
                len(active_bugs)
            ),
            "eligible": (
                len(eligible)
            ),
            "selected": (
                len(selected)
            ),
            "selected_bug_ids": (
                selected
            ),
        }

        for bug_id in selected:

            selected_entries.append(
                {
                    "benchmark_id": (
                        f"{project}-{bug_id}"
                    ),
                    "project": (
                        project
                    ),
                    "bug_id": (
                        bug_id
                    ),
                }
            )

    payload = {
        "name": (
            "CAMD Retriever Development Set v1"
        ),
        "purpose": (
            "Development-only benchmark for "
            "non-oracle program-wide method "
            "retrieval. Must not overlap with "
            "the frozen FSE/ASE final benchmark."
        ),
        "seed": (
            SEED
        ),
        "bugs_per_project": (
            PER_PROJECT
        ),
        "projects": (
            PROJECTS
        ),
        "selection_policy": {
            "exclude_final_benchmark": True,
            "exclude_lang_1_20": True,
            "performance_based_selection": (
                False
            ),
        },
        "total_selected": (
            len(
                selected_entries
            )
        ),
        "project_summary": (
            project_summary
        ),
        "entries": (
            selected_entries
        ),
    }

    save_json(
        OUTPUT_FILE,
        payload,
    )

    print()
    print("=" * 100)
    print(
        "CAMD Retriever Development Set"
    )
    print("=" * 100)

    print(
        f"Seed: {SEED}"
    )

    print(
        f"Per project: "
        f"{PER_PROJECT}"
    )

    print(
        f"Total selected: "
        f"{len(selected_entries)}"
    )

    print()

    for project in PROJECTS:

        info = (
            project_summary[
                project
            ]
        )

        print(
            f"{project}: "
            f"{info['selected_bug_ids']}"
        )

    print()
    print(
        f"Saved:\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
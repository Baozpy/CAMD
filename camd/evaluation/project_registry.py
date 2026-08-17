from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Defects4JProjectConfig:
    project_id: str
    display_name: str
    max_bug_id: int
    deprecated_bug_ids: frozenset[int]

    def is_valid_bug(self, bug_id: int) -> bool:
        return (
            1 <= bug_id <= self.max_bug_id
            and bug_id not in self.deprecated_bug_ids
        )

    def valid_bug_ids(
        self,
        start: int | None = None,
        end: int | None = None,
    ) -> list[int]:
        lo = 1 if start is None else start
        hi = self.max_bug_id if end is None else min(
            end,
            self.max_bug_id,
        )

        return [
            bug_id
            for bug_id in range(lo, hi + 1)
            if self.is_valid_bug(bug_id)
        ]


PROJECTS: dict[str, Defects4JProjectConfig] = {
    "Lang": Defects4JProjectConfig(
        project_id="Lang",
        display_name="Apache Commons Lang",
        max_bug_id=65,
        deprecated_bug_ids=frozenset({
            2,
            18,
        }),
    ),

    "Math": Defects4JProjectConfig(
        project_id="Math",
        display_name="Apache Commons Math",
        max_bug_id=106,
        deprecated_bug_ids=frozenset(),
    ),

    "Chart": Defects4JProjectConfig(
        project_id="Chart",
        display_name="JFreeChart",
        max_bug_id=26,
        deprecated_bug_ids=frozenset(),
    ),

    "Time": Defects4JProjectConfig(
        project_id="Time",
        display_name="Joda-Time",
        max_bug_id=27,
        deprecated_bug_ids=frozenset(),
    ),

    "Mockito": Defects4JProjectConfig(
        project_id="Mockito",
        display_name="Mockito",
        max_bug_id=38,
        deprecated_bug_ids=frozenset(),
    ),
}


def get_project_config(
    project_id: str,
) -> Defects4JProjectConfig:
    try:
        return PROJECTS[project_id]
    except KeyError as exc:
        supported = ", ".join(
            sorted(PROJECTS)
        )

        raise ValueError(
            f"Unsupported Defects4J project: "
            f"{project_id}. "
            f"Supported projects: {supported}"
        ) from exc


def list_supported_projects() -> list[str]:
    return sorted(
        PROJECTS.keys()
    )
from dataclasses import dataclass

from camd.evaluation.statement_ground_truth import (
    StatementRegion,
)


@dataclass
class RankingMetrics:
    first_hit_rank: int | None

    top_1: bool
    top_3: bool
    top_5: bool
    top_10: bool

    reciprocal_rank: float


@dataclass
class LineLocalizationMetrics:
    ground_truth_lines: list[int]
    predicted_lines: list[int]

    statement_regions: list[dict]

    exact: RankingMetrics
    statement: RankingMetrics
    region_2: RankingMetrics


def _build_metrics(
    first_hit_rank: int | None,
) -> RankingMetrics:

    if first_hit_rank is None:

        return RankingMetrics(
            first_hit_rank=None,
            top_1=False,
            top_3=False,
            top_5=False,
            top_10=False,
            reciprocal_rank=0.0,
        )

    return RankingMetrics(
        first_hit_rank=(
            first_hit_rank
        ),

        top_1=(
            first_hit_rank <= 1
        ),

        top_3=(
            first_hit_rank <= 3
        ),

        top_5=(
            first_hit_rank <= 5
        ),

        top_10=(
            first_hit_rank <= 10
        ),

        reciprocal_rank=(
            1.0
            / first_hit_rank
        ),
    )


def _find_exact_hit(
    predicted_lines: list[int],
    ground_truth_lines: list[int],
) -> int | None:

    gt_set = set(
        ground_truth_lines
    )

    for rank, line in enumerate(
        predicted_lines,
        start=1,
    ):

        if line in gt_set:
            return rank

    return None


def _find_statement_hit(
    predicted_lines: list[int],
    statement_regions: list[
        StatementRegion
    ],
) -> int | None:

    for rank, line in enumerate(
        predicted_lines,
        start=1,
    ):

        for region in statement_regions:

            if (
                region.start_line
                <= line
                <= region.end_line
            ):
                return rank

    return None


def _find_region_hit(
    predicted_lines: list[int],
    ground_truth_lines: list[int],
    radius: int = 2,
) -> int | None:

    for rank, predicted in enumerate(
        predicted_lines,
        start=1,
    ):

        for gt in ground_truth_lines:

            if (
                abs(
                    predicted - gt
                )
                <= radius
            ):
                return rank

    return None


def evaluate_line_ranking(
    predicted_lines: list[int],
    ground_truth_lines: list[int],
    statement_regions: list[
        StatementRegion
    ] | None = None,
) -> LineLocalizationMetrics:

    if statement_regions is None:
        statement_regions = []

    exact_rank = (
        _find_exact_hit(
            predicted_lines=(
                predicted_lines
            ),
            ground_truth_lines=(
                ground_truth_lines
            ),
        )
    )

    statement_rank = (
        _find_statement_hit(
            predicted_lines=(
                predicted_lines
            ),
            statement_regions=(
                statement_regions
            ),
        )
    )

    region_rank = (
        _find_region_hit(
            predicted_lines=(
                predicted_lines
            ),
            ground_truth_lines=(
                ground_truth_lines
            ),
            radius=2,
        )
    )

    return LineLocalizationMetrics(
        ground_truth_lines=(
            sorted(
                set(
                    ground_truth_lines
                )
            )
        ),

        predicted_lines=(
            predicted_lines
        ),

        statement_regions=[
            {
                "start_line": (
                    region.start_line
                ),
                "end_line": (
                    region.end_line
                ),
                "node_type": (
                    region.node_type
                ),
                "anchor_lines": (
                    region.anchor_lines
                ),
            }
            for region
            in statement_regions
        ],

        exact=(
            _build_metrics(
                exact_rank
            )
        ),

        statement=(
            _build_metrics(
                statement_rank
            )
        ),

        region_2=(
            _build_metrics(
                region_rank
            )
        ),
    )
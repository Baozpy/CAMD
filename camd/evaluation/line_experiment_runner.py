import json
from dataclasses import asdict
from pathlib import Path

from camd.context.method_extractor import extract_java_methods
from camd.detectors.line_ranker import LineRanker
from camd.evaluation.diff_ground_truth import extract_changed_ranges
from camd.evaluation.failing_test_extractor import FailingTestExtractor
from camd.evaluation.line_evaluator import evaluate_line_ranking
from camd.evaluation.test_context_builder import TestContextBuilder
from camd.llm.client import OpenAIClient
from camd.static.ast_analyzer import JavaASTAnalyzer
from camd.static.evidence_builder import StaticEvidenceBuilder
from camd.evaluation.statement_ground_truth import (
    StatementGroundTruthBuilder,
)

class LineExperimentRunner:

    def __init__(
        self,
        project_root: Path,
        top_k: int = 10,
    ):
        self.project_root = project_root
        self.results_root = (
            project_root
            / "results"
            / "defects4j"
        )

        self.checkouts_root = (
            project_root
            / "data"
            / "defects4j"
            / "checkouts"
        )

        self.top_k = top_k

        self.llm_client = OpenAIClient()

        self.line_ranker = LineRanker(
            llm_client=self.llm_client
        )

        self.ast_analyzer = JavaASTAnalyzer()
        self.evidence_builder = StaticEvidenceBuilder()
        self.test_context_builder = TestContextBuilder()
        self.statement_gt_builder = (
            StatementGroundTruthBuilder(
                anchor_tolerance=2
            )
        )

    @staticmethod
    def _add_line_numbers(
        code: str,
        start_line: int,
    ) -> str:

        output = []

        for offset, line in enumerate(
            code.splitlines()
        ):
            absolute_line = (
                start_line + offset
            )

            output.append(
                f"{absolute_line:5d}: {line}"
            )

        return "\n".join(output)

    @staticmethod
    def _load_json(
        path: Path,
    ) -> dict:

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    @staticmethod
    def _save_json(
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

    def _build_failing_test_context(
        self,
        checkout_dir: Path,
    ) -> str:

        extractor = FailingTestExtractor(
            checkout_dir=checkout_dir
        )

        tests = extractor.extract()

        parts = []

        for test in tests:

            parts.append(
                f"Failing test: {test.full_name}"
            )
            parts.append("")

            expanded = False

            if (
                test.source_file is not None
                and test.method_name
            ):
                try:
                    context = (
                        self.test_context_builder.build(
                            source_file=test.source_file,
                            test_method_name=test.method_name,
                        )
                    )

                    parts.append(
                        context.to_text()
                    )

                    expanded = True

                except ValueError:
                    pass

            if (
                not expanded
                and test.code
            ):
                parts.append(test.code)

            parts.append("")
            parts.append("=" * 80)

        return "\n".join(parts)

    @staticmethod
    def _find_source_file(
        checkout_dir: Path,
        modified_class: str,
    ) -> Path | None:

        relative = Path(
            *modified_class.split(".")
        ).with_suffix(".java")

        possible_roots = [
            checkout_dir
            / "src"
            / "main"
            / "java",

            checkout_dir
            / "src"
            / "java",

            checkout_dir
            / "src",
        ]

        for root in possible_roots:
            candidate = root / relative

            if candidate.exists():
                return candidate

        filename = (
            modified_class.split(".")[-1]
            + ".java"
        )

        matches = list(
            checkout_dir.rglob(filename)
        )

        matches = [
            path
            for path in matches
            if (
                "test" not in {
                    part.lower()
                    for part in path.parts
                }
            )
        ]

        if len(matches) == 1:
            return matches[0]

        return None

    @staticmethod
    def _ground_truth_lines_for_method(
        changed_ranges,
        method_start: int,
        method_end: int,
    ) -> list[int]:

        lines = set()

        for changed in changed_ranges:

            start = max(
                changed.start_line,
                method_start,
            )

            end = min(
                changed.end_line,
                method_end,
            )

            if start > end:
                continue

            for line in range(
                start,
                end + 1,
            ):
                lines.add(line)

        return sorted(lines)

    def run_oracle_bug(
        self,
        bug_record: dict,
    ) -> dict:

        project = bug_record["project"]
        bug_id = bug_record["bug_id"]

        buggy_dir = (
            self.checkouts_root
            / f"{project}_{bug_id}b"
        )

        fixed_dir = (
            self.checkouts_root
            / f"{project}_{bug_id}f"
        )

        if not buggy_dir.exists():
            raise RuntimeError(
                f"Buggy checkout missing: "
                f"{buggy_dir}"
            )

        if not fixed_dir.exists():
            raise RuntimeError(
                f"Fixed checkout missing: "
                f"{fixed_dir}"
            )

        failing_test_context = (
            self._build_failing_test_context(
                buggy_dir
            )
        )

        all_method_results = []

        for modified_class in bug_record.get(
            "modified_classes",
            [],
        ):
            buggy_file = (
                self._find_source_file(
                    buggy_dir,
                    modified_class,
                )
            )

            fixed_file = (
                self._find_source_file(
                    fixed_dir,
                    modified_class,
                )
            )

            if (
                buggy_file is None
                or fixed_file is None
            ):
                continue

            methods = extract_java_methods(
                buggy_file
            )

            changed_ranges = (
                extract_changed_ranges(
                    buggy_file=buggy_file,
                    fixed_file=fixed_file,
                )
            )

            gt_methods = [
                gt
                for gt in bug_record.get(
                    "ground_truth_methods",
                    []
                )
            ]

            for gt in gt_methods:

                method_name = gt["name"]

                candidates = [
                    method
                    for method in methods
                    if (
                        method.name
                        == method_name
                    )
                ]

                if not candidates:
                    continue

                # Resolve overloaded methods using overlap
                # with the saved ground-truth range.
                gt_start = gt.get(
                    "start_line"
                )
                gt_end = gt.get(
                    "end_line"
                )

                if (
                    gt_start is not None
                    and gt_end is not None
                ):
                    overlapping = [
                        method
                        for method in candidates
                        if not (
                            method.end_line
                            < gt_start
                            or method.start_line
                            > gt_end
                        )
                    ]

                    if overlapping:
                        candidates = overlapping

                for method in candidates:

                    gt_lines = (
                        self._ground_truth_lines_for_method(
                            changed_ranges=changed_ranges,
                            method_start=method.start_line,
                            method_end=method.end_line,
                        )
                    )

                    if not gt_lines:
                        continue

                    method_code = (
                        self._add_line_numbers(
                            code=method.code,
                            start_line=method.start_line,
                        )
                    )

                    static_evidence = (
                        self.ast_analyzer.analyze(
                            method
                        )
                    )

                    static_text = (
                        self.evidence_builder.build_text(
                            static_evidence
                        )
                    )

                    predictions = (
                        self.line_ranker.rank(
                            method_name=method.name,
                            method_code_with_lines=method_code,
                            static_evidence=static_text,
                            failing_test_context=(
                                failing_test_context
                            ),
                            top_k=self.top_k,
                        )
                    )

                    predicted_lines = [
                        prediction.line
                        for prediction in predictions
                    ]

                    statement_regions = (
                        self.statement_gt_builder.build(
                            source_file=buggy_file,
                            ground_truth_lines=gt_lines,
                        )
                    )

                    metrics = (
                        evaluate_line_ranking(
                            predicted_lines=(
                                predicted_lines
                            ),
                            ground_truth_lines=(
                                gt_lines
                            ),
                            statement_regions=(
                                statement_regions
                            ),
                        )
                    )

                    all_method_results.append(
                        {
                            "class_name": (
                                modified_class
                            ),
                            "method_name": (
                                method.name
                            ),
                            "method_start_line": (
                                method.start_line
                            ),
                            "method_end_line": (
                                method.end_line
                            ),
                            "ground_truth_lines": (
                                gt_lines
                            ),
                            "predictions": [
                                asdict(prediction)
                                for prediction
                                in predictions
                            ],
                            "metrics": (
                                asdict(metrics)
                            ),
                        }
                    )

        if not all_method_results:
            raise RuntimeError(
                "No oracle ground-truth method "
                "could be evaluated."
            )

        # A Defects4J bug can contain multiple changed
        # methods. For bug-level localization we use the
        # best first-hit rank among valid GT methods,
        # consistent with the existing method-level
        # evaluation protocol.
        successful = [
            item
            for item in all_method_results
            if (
                 item["metrics"][
                    "statement"
                ][
                    "first_hit_rank"
                ]
                is not None
            )
        ]

        if successful:
            best = min(
                successful,
                key=lambda item: (
                    item["metrics"][
                        "statement"
                    ][
                        "first_hit_rank"
                    ]
                ),
            )

            bug_metrics = dict(
                best["metrics"]
            )

            best_method = {
                "class_name": (
                    best["class_name"]
                ),
                "method_name": (
                    best["method_name"]
                ),
                "method_start_line": (
                    best["method_start_line"]
                ),
                "method_end_line": (
                    best["method_end_line"]
                ),
            }

        else:
            bug_metrics = {
                "ground_truth_lines": [],
                "predicted_lines": [],
                "statement_regions": [],

                "exact": {
                    "first_hit_rank": None,
                    "top_1": False,
                    "top_3": False,
                    "top_5": False,
                    "top_10": False,
                    "reciprocal_rank": 0.0,
                },

                "statement": {
                    "first_hit_rank": None,
                    "top_1": False,
                    "top_3": False,
                    "top_5": False,
                    "top_10": False,
                    "reciprocal_rank": 0.0,
                },

                "region_2": {
                    "first_hit_rank": None,
                    "top_1": False,
                    "top_3": False,
                    "top_5": False,
                    "top_10": False,
                    "reciprocal_rank": 0.0,
                },
            }

            best_method = None

        result = {
            "project": project,
            "bug_id": bug_id,
            "mode": "oracle_method",
            "best_ground_truth_method": (
                best_method
            ),
            "metrics": bug_metrics,
            "ground_truth_method_results": (
                all_method_results
            ),
        }

        output_file = (
            self.results_root
            / f"{project}_{bug_id}"
            / "line_localization_oracle.json"
        )

        self._save_json(
            output_file,
            result,
        )

        return result
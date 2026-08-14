import json
from dataclasses import asdict
from pathlib import Path

from camd.context.method_extractor import extract_java_methods
from camd.detectors.line_ranker import LineRanker
from camd.evaluation.diff_ground_truth import extract_changed_ranges
from camd.evaluation.failing_test_extractor import FailingTestExtractor
from camd.evaluation.line_evaluator import evaluate_line_ranking
from camd.evaluation.statement_ground_truth import StatementGroundTruthBuilder
from camd.evaluation.test_context_builder import TestContextBuilder
from camd.llm.client import OpenAIClient
from camd.static.ast_analyzer import JavaASTAnalyzer
from camd.static.evidence_builder import StaticEvidenceBuilder


class EndToEndLineRunner:

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
                start_line
                + offset
            )

            output.append(
                f"{absolute_line:5d}: {line}"
            )

        return "\n".join(output)

    @staticmethod
    def _find_source_file(
        checkout_dir: Path,
        class_name: str,
    ) -> Path | None:

        relative = Path(
            *class_name.split(".")
        ).with_suffix(".java")

        roots = [
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

        for root in roots:

            candidate = (
                root
                / relative
            )

            if candidate.exists():
                return candidate

        filename = (
            class_name.split(".")[-1]
            + ".java"
        )

        matches = list(
            checkout_dir.rglob(
                filename
            )
        )

        matches = [
            path
            for path in matches
            if (
                "test"
                not in {
                    part.lower()
                    for part in path.parts
                }
            )
        ]

        if len(matches) == 1:
            return matches[0]

        return None

    def _build_failing_test_context(
        self,
        checkout_dir: Path,
    ) -> str:

        extractor = (
            FailingTestExtractor(
                checkout_dir=checkout_dir
            )
        )

        tests = extractor.extract()

        parts = []

        for test in tests:

            parts.append(
                f"Failing test: "
                f"{test.full_name}"
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
                            source_file=(
                                test.source_file
                            ),
                            test_method_name=(
                                test.method_name
                            ),
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

                parts.append(
                    test.code
                )

            parts.append("")
            parts.append(
                "=" * 80
            )

        return "\n".join(
            parts
        )

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
                lines.add(
                    line
                )

        return sorted(
            lines
        )

    @staticmethod
    def _method_matches_ground_truth(
        class_name: str,
        method,
        ground_truth_methods: list[dict],
    ) -> bool:

        for gt in ground_truth_methods:

            if (
                gt.get("name")
                != method.name
            ):
                continue

            gt_start = gt.get(
                "start_line"
            )

            gt_end = gt.get(
                "end_line"
            )

            if (
                gt_start is None
                or gt_end is None
            ):
                return True

            overlaps = not (
                method.end_line < gt_start
                or method.start_line > gt_end
            )

            if overlaps:
                return True

        return False

    def _load_camd_top1(
        self,
        project: str,
        bug_id: int,
    ) -> dict | None:

        bug_dir = (
            self.results_root
            / f"{project}_{bug_id}"
        )

        candidate_files = [
            bug_dir
            / "multi_agent_expanded_test.jsonl",

            bug_dir
            / "multi_agent.jsonl",
        ]

        for path in candidate_files:

            if not path.exists():
                continue

            records = []

            with path.open(
                "r",
                encoding="utf-8",
            ) as file:

                for line in file:

                    line = line.strip()

                    if not line:
                        continue

                    records.append(
                        json.loads(line)
                    )

            if not records:
                continue

            def get_score(
                item: dict,
            ) -> float:

                judge = item.get(
                    "judge"
                )

                if isinstance(
                    judge,
                    dict,
                ):
                    return float(
                        judge.get(
                            "target_defect_probability",
                            0.0,
                        )
                    )

                return 0.0

            ranked = sorted(
                records,
                key=get_score,
                reverse=True,
            )

            top = ranked[0]

            method_name = top.get(
                "method"
            )

            if method_name is None:
                continue

            return {
                "method_name": (
                    method_name
                ),

                "start_line": (
                    top.get(
                        "start_line"
                    )
                ),

                "end_line": (
                    top.get(
                        "end_line"
                    )
                ),

                "b4_rank": (
                    top.get(
                        "b4_rank"
                    )
                ),

                "b4_score": (
                    top.get(
                        "b4_score"
                    )
                ),

                "judge_score": (
                    get_score(top)
                ),

                "source_file": (
                    path.name
                ),
            }

        return None


    def run_bug(
        self,
        bug_record: dict,
    ) -> dict:

        project = (
            bug_record[
                "project"
            ]
        )

        bug_id = (
            bug_record[
                "bug_id"
            ]
        )

        buggy_dir = (
            self.checkouts_root
            / f"{project}_{bug_id}b"
        )

        fixed_dir = (
            self.checkouts_root
            / f"{project}_{bug_id}f"
        )

        selected = (
            self._load_camd_top1(
                project=project,
                bug_id=bug_id,
            )
        )

        if selected is None:

            return {
                "project": project,
                "bug_id": bug_id,
                "selected_method": None,
                "method_correct": False,
                "line_localization_attempted": False,
                "failure_reason": (
                    "No saved CAMD Top-1 "
                    "method prediction found."
                ),
                "metrics": self._empty_metrics(),
            }

        selected_name = (
            selected[
                "method_name"
            ]
        )

        selected_class = (
            selected.get(
                "class_name"
            )
        )

        modified_classes = (
            bug_record.get(
                "modified_classes",
                [],
            )
        )

        if (
            selected_class
            not in modified_classes
        ):
            selected_class = None

        source_candidates = []

        for class_name in (
            modified_classes
        ):

            if (
                selected_class is not None
                and class_name
                != selected_class
            ):
                continue

            buggy_file = (
                self._find_source_file(
                    buggy_dir,
                    class_name,
                )
            )

            fixed_file = (
                self._find_source_file(
                    fixed_dir,
                    class_name,
                )
            )

            if (
                buggy_file is None
                or fixed_file is None
            ):
                continue

            methods = (
                extract_java_methods(
                    buggy_file
                )
            )

            matches = [
                method
                for method in methods
                if (
                    method.name
                    == selected_name
                )
            ]

            for method in matches:

                source_candidates.append(
                    (
                        class_name,
                        buggy_file,
                        fixed_file,
                        method,
                    )
                )

        if not source_candidates:

            return {
                "project": project,
                "bug_id": bug_id,
                "selected_method": selected,
                "method_correct": False,
                "line_localization_attempted": False,
                "failure_reason": (
                    "Saved CAMD method could "
                    "not be resolved in buggy "
                    "source."
                ),
                "metrics": self._empty_metrics(),
            }

        # if len(source_candidates) > 1:

        #     gt_methods = (
        #         bug_record.get(
        #             "ground_truth_methods",
        #             [],
        #         )
        #     )

        #     gt_matching = [
        #         item
        #         for item
        #         in source_candidates
        #         if self._method_matches_ground_truth(
        #             class_name=item[0],
        #             method=item[3],
        #             ground_truth_methods=(
        #                 gt_methods
        #             ),
        #         )
        #     ]

        #     if len(gt_matching) == 1:
        #         source_candidates = (
        #             gt_matching
        #         )
        selected_start = (
            selected.get(
                "start_line"
            )
        )

        selected_end = (
            selected.get(
                "end_line"
            )
        )

        if (
            len(source_candidates) > 1
            and selected_start is not None
            and selected_end is not None
        ):

            exact_range_matches = [
                item
                for item in source_candidates
                if (
                    item[3].start_line
                    == selected_start
                    and item[3].end_line
                    == selected_end
                )
            ]

            if len(exact_range_matches) == 1:

                source_candidates = (
                    exact_range_matches
                )

            else:

                overlap_matches = [
                    item
                    for item in source_candidates
                    if not (
                        item[3].end_line
                        < selected_start
                        or item[3].start_line
                        > selected_end
                    )
                ]

                if len(overlap_matches) == 1:

                    source_candidates = (
                        overlap_matches
                    )

        if len(source_candidates) != 1:

            return {
                "project": project,
                "bug_id": bug_id,

                "selected_method": (
                    selected
                ),

                "method_correct": False,

                "line_localization_attempted": (
                    False
                ),

                "failure_reason": (
                    "Saved CAMD Top-1 method "
                    "could not be uniquely resolved "
                    "using its saved source range."
                ),

                "metrics": (
                    self._empty_metrics()
                ),
            }

#########################
        class_name, buggy_file, fixed_file, method = (
            source_candidates[0]
        )

        method_correct = (
            self._method_matches_ground_truth(
                class_name=class_name,
                method=method,
                ground_truth_methods=(
                    bug_record.get(
                        "ground_truth_methods",
                        [],
                    )
                ),
            )
        )

        if not method_correct:

            return {
                "project": project,
                "bug_id": bug_id,

                "selected_method": {
                    **selected,
                    "resolved_class_name": (
                        class_name
                    ),
                    "start_line": (
                        method.start_line
                    ),
                    "end_line": (
                        method.end_line
                    ),
                },

                "method_correct": False,

                "line_localization_attempted": (
                    False
                ),

                "failure_reason": (
                    "CAMD Top-1 method is not "
                    "a ground-truth method."
                ),

                "metrics": (
                    self._empty_metrics()
                ),
            }

        changed_ranges = (
            extract_changed_ranges(
                buggy_file=buggy_file,
                fixed_file=fixed_file,
            )
        )

        gt_lines = (
            self._ground_truth_lines_for_method(
                changed_ranges=changed_ranges,
                method_start=method.start_line,
                method_end=method.end_line,
            )
        )

        if not gt_lines:

            return {
                "project": project,
                "bug_id": bug_id,
                "selected_method": selected,
                "method_correct": True,
                "line_localization_attempted": False,
                "failure_reason": (
                    "Ground-truth changed lines "
                    "could not be mapped into "
                    "selected method."
                ),
                "metrics": self._empty_metrics(),
            }

        failing_test_context = (
            self._build_failing_test_context(
                buggy_dir
            )
        )

        numbered_code = (
            self._add_line_numbers(
                code=method.code,
                start_line=(
                    method.start_line
                ),
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
                method_name=(
                    method.name
                ),

                method_code_with_lines=(
                    numbered_code
                ),

                static_evidence=(
                    static_text
                ),

                failing_test_context=(
                    failing_test_context
                ),

                top_k=(
                    self.top_k
                ),
            )
        )

        predicted_lines = [
            item.line
            for item
            in predictions
        ]

        statement_regions = (
            self.statement_gt_builder.build(
                source_file=buggy_file,
                ground_truth_lines=(
                    gt_lines
                ),
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

        result = {
            "project": project,
            "bug_id": bug_id,

            "selected_method": {
                **selected,

                "resolved_class_name": (
                    class_name
                ),

                "start_line": (
                    method.start_line
                ),

                "end_line": (
                    method.end_line
                ),
            },

            "method_correct": True,

            "line_localization_attempted": (
                True
            ),

            "ground_truth_lines": (
                gt_lines
            ),

            "predictions": [
                asdict(item)
                for item
                in predictions
            ],

            "metrics": (
                asdict(metrics)
            ),
        }

        output_file = (
            self.results_root
            / f"{project}_{bug_id}"
            / "line_localization_end_to_end.json"
        )

        self._save_json(
            output_file,
            result,
        )

        return result

    @staticmethod
    def _empty_ranking_metrics() -> dict:

        return {
            "first_hit_rank": None,
            "top_1": False,
            "top_3": False,
            "top_5": False,
            "top_10": False,
            "reciprocal_rank": 0.0,
        }

    @classmethod
    def _empty_metrics(
        cls,
    ) -> dict:

        return {
            "ground_truth_lines": [],
            "predicted_lines": [],
            "statement_regions": [],

            "exact": (
                cls._empty_ranking_metrics()
            ),

            "statement": (
                cls._empty_ranking_metrics()
            ),

            "region_2": (
                cls._empty_ranking_metrics()
            ),
        }
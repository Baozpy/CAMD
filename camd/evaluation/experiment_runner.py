import json
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

from camd.agents.critic_agent import CriticAgent
from camd.agents.detector_agent import DetectorAgent
from camd.agents.judge_agent import JudgeAgent

from camd.context.method_extractor import (
    JavaMethod,
    extract_java_methods,
)
from camd.context.semantic_context_builder import (
    SemanticContextBuilder,
)

from camd.detectors.method_ranker import (
    MethodRanker,
)
from camd.detectors.static_context_ranker import (
    StaticContextRanker,
    format_semantic_context,
)

from camd.evaluation.diff_ground_truth import (
    extract_changed_ranges,
)
from camd.evaluation.failing_test_extractor import (
    FailingTestExtractor,
)

from camd.llm.client import OpenAIClient

from camd.static.ast_analyzer import (
    JavaASTAnalyzer,
)
from camd.static.evidence_builder import (
    StaticEvidenceBuilder,
)


class Defects4JExperimentRunner:

    def __init__(
        self,
        project_root: Path,
        project: str,
        bug_id: int,
        top_k: int = 5,
    ):
        self.project_root = project_root
        self.project = project
        self.bug_id = bug_id
        self.top_k = top_k

        self.checkouts_root = (
            project_root
            / "data"
            / "defects4j"
            / "checkouts"
        )

        self.buggy_dir = (
            self.checkouts_root
            / f"{project}_{bug_id}b"
        )

        self.fixed_dir = (
            self.checkouts_root
            / f"{project}_{bug_id}f"
        )

        self.result_dir = (
            project_root
            / "results"
            / "defects4j"
            / f"{project}_{bug_id}"
        )

        self.result_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.client = OpenAIClient()

    # =========================================================
    # Defects4J utilities
    # =========================================================

    def _run_command(
        self,
        command: list[str],
        cwd: Path | None = None,
    ) -> str:

        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:

            raise RuntimeError(
                "Command failed:\n"
                f"{' '.join(command)}\n\n"
                f"STDOUT:\n{result.stdout}\n\n"
                f"STDERR:\n{result.stderr}"
            )

        return result.stdout.strip()

    def ensure_checkout(
        self,
        version: str,
        directory: Path,
    ) -> None:

        if directory.exists():
            print(
                f"Using existing checkout: "
                f"{directory.name}"
            )
            return

        directory.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(
            f"Checking out "
            f"{self.project}-{self.bug_id}{version}..."
        )

        self._run_command(
            [
                "defects4j",
                "checkout",
                "-p",
                self.project,
                "-v",
                f"{self.bug_id}{version}",
                "-w",
                str(directory),
            ]
        )

    def prepare_checkouts(
        self,
    ) -> None:

        self.ensure_checkout(
            version="b",
            directory=self.buggy_dir,
        )

        self.ensure_checkout(
            version="f",
            directory=self.fixed_dir,
        )

    def export_property(
        self,
        checkout_dir: Path,
        property_name: str,
    ) -> str:

        return self._run_command(
            [
                "defects4j",
                "export",
                "-p",
                property_name,
            ],
            cwd=checkout_dir,
        )

    # =========================================================
    # Source discovery
    # =========================================================

    def get_modified_classes(
        self,
    ) -> list[str]:

        output = self.export_property(
            self.buggy_dir,
            "classes.modified",
        )

        classes = [
            line.strip()
            for line in output.splitlines()
            if line.strip()
        ]

        return classes

    def get_source_root(
        self,
        checkout_dir: Path,
    ) -> Path:

        source_dir = self.export_property(
            checkout_dir,
            "dir.src.classes",
        )

        return (
            checkout_dir
            / source_dir
        )

    def class_to_source_file(
        self,
        checkout_dir: Path,
        class_name: str,
    ) -> Path:

        source_root = self.get_source_root(
            checkout_dir
        )

        relative = Path(
            *class_name.split(".")
        ).with_suffix(
            ".java"
        )

        return source_root / relative

    # =========================================================
    # Ground truth
    # =========================================================

    def get_ground_truth_methods(
        self,
        buggy_file: Path,
        fixed_file: Path,
        buggy_methods: list[JavaMethod],
    ) -> list[JavaMethod]:

        changed_ranges = (
            extract_changed_ranges(
                buggy_file=buggy_file,
                fixed_file=fixed_file,
            )
        )

        ground_truth = []

        seen = set()

        for changed in changed_ranges:

            for method in buggy_methods:

                overlaps = not (
                    method.end_line
                    < changed.start_line
                    or method.start_line
                    > changed.end_line
                )

                if not overlaps:
                    continue

                key = (
                    method.name,
                    method.start_line,
                    method.end_line,
                )

                if key in seen:
                    continue

                seen.add(key)

                ground_truth.append(
                    method
                )

        return ground_truth

    # =========================================================
    # Ranking metrics
    # =========================================================

    @staticmethod
    def find_best_ground_truth_rank(
        results,
        ground_truth_methods: list[JavaMethod],
    ) -> float | None:

        gt_keys = {
            (
                method.name,
                method.start_line,
                method.end_line,
            )
            for method in ground_truth_methods
        }

        gt_results = [
            result
            for result in results
            if (
                result.method_name,
                result.start_line,
                result.end_line,
            )
            in gt_keys
        ]

        if not gt_results:
            return None

        best_rank = None

        for gt_result in gt_results:

            gt_score = (
                gt_result.suspicion_score
            )

            strictly_higher = sum(
                1
                for result in results
                if (
                    result.suspicion_score
                    > gt_score
                )
            )

            tied = sum(
                1
                for result in results
                if (
                    result.suspicion_score
                    == gt_score
                )
            )

            average_rank = (
                strictly_higher
                + 1
                + (tied - 1) / 2
            )

            if (
                best_rank is None
                or average_rank < best_rank
            ):
                best_rank = average_rank

        return best_rank

    @staticmethod
    def reciprocal_rank(
        rank: float | None,
    ) -> float:

        if rank is None:
            return 0.0

        return 1.0 / rank

    @staticmethod
    def build_metrics(
        rank: float | None,
    ) -> dict:

        if rank is None:

            return {
                "rank": None,
                "rr": 0.0,
                "top_1": False,
                "top_3": False,
                "top_5": False,
                "top_10": False,
            }

        return {
            "rank": rank,
            "rr": 1.0 / rank,
            "top_1": rank <= 1,
            "top_3": rank <= 3,
            "top_5": rank <= 5,
            "top_10": rank <= 10,
        }

    # =========================================================
    # Result writing
    # =========================================================

    @staticmethod
    def write_jsonl(
        path: Path,
        records: list[dict],
    ) -> None:

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with path.open(
            "w",
            encoding="utf-8",
        ) as file:

            for record in records:

                file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                )

                file.write("\n")

    @staticmethod
    def write_json(
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

    # =========================================================
    # B1
    # =========================================================

    def run_b1(
        self,
        methods: list[JavaMethod],
    ):

        print(
            "\nRunning B1: "
            "Method-Only Ranking..."
        )

        ranker = MethodRanker(
            llm_client=self.client
        )

        return ranker.rank_methods(
            methods
        )

    # =========================================================
    # B4
    # =========================================================

    def run_b4(
        self,
        methods: list[JavaMethod],
    ):

        print(
            "\nRunning B4: "
            "Static-Aware Ranking..."
        )

        ranker = StaticContextRanker(
            llm_client=self.client,
            methods=methods,
            top_k_callees=3,
            top_k_callers=2,
        )

        return ranker.rank_methods(
            methods
        )

    # =========================================================
    # Failing test context
    # =========================================================

    def build_failing_test_context(
        self,
    ) -> tuple[str, list[str]]:

        extractor = (
            FailingTestExtractor(
                checkout_dir=self.buggy_dir
            )
        )

        tests = extractor.extract()

        output = []
        names = []

        for test in tests:

            names.append(
                test.full_name
            )

            output.append(
                f"Test: {test.full_name}"
            )

            if (
                test.start_line is not None
                and test.end_line is not None
            ):

                output.append(
                    f"Lines: "
                    f"{test.start_line}-"
                    f"{test.end_line}"
                )

            output.append("")

            if test.code:
                output.append(
                    test.code
                )

            output.append("")
            output.append(
                "=" * 70
            )

        return (
            "\n".join(output),
            names,
        )

    # =========================================================
    # Candidate context
    # =========================================================

    def build_candidate_context(
        self,
        method: JavaMethod,
        methods: list[JavaMethod],
    ) -> str:

        semantic_builder = (
            SemanticContextBuilder(
                methods=methods,
                top_k_callees=3,
                top_k_callers=2,
            )
        )

        ast_analyzer = (
            JavaASTAnalyzer()
        )

        evidence_builder = (
            StaticEvidenceBuilder()
        )

        semantic_context = (
            semantic_builder.build(
                method
            )
        )

        semantic_text = (
            format_semantic_context(
                semantic_context
            )
        )

        static_evidence = (
            ast_analyzer.analyze(
                method
            )
        )

        static_text = (
            evidence_builder.build_text(
                static_evidence
            )
        )

        return f"""
SEMANTIC CONTEXT
================

{semantic_text}


STATIC EVIDENCE
===============

{static_text}
""".strip()

    # =========================================================
    # Multi-Agent
    # =========================================================

    def run_multi_agent(
        self,
        methods: list[JavaMethod],
        b4_results,
        failing_test_context: str,
    ) -> list[dict]:

        print(
            "\nRunning CAMD "
            "Multi-Agent verification..."
        )

        detector = DetectorAgent(
            llm_client=self.client
        )

        critic = CriticAgent(
            llm_client=self.client
        )

        judge = JudgeAgent(
            llm_client=self.client
        )

        top_candidates = (
            b4_results[
                : self.top_k
            ]
        )

        results = []

        for index, candidate in enumerate(
            top_candidates,
            start=1,
        ):

            print(
                f"\n[{index}/{len(top_candidates)}] "
                f"{candidate.method_name} "
                f"({candidate.start_line}-"
                f"{candidate.end_line})"
            )

            method = next(
                (
                    method
                    for method in methods
                    if (
                        method.name
                        == candidate.method_name
                        and method.start_line
                        == candidate.start_line
                        and method.end_line
                        == candidate.end_line
                    )
                ),
                None,
            )

            if method is None:
                continue

            candidate_context = (
                self.build_candidate_context(
                    method=method,
                    methods=methods,
                )
            )

            detector_result = (
                detector.analyze(
                    method_name=method.name,
                    candidate_context=(
                        candidate_context
                    ),
                    failing_test_context=(
                        failing_test_context
                    ),
                )
            )

            print(
                "  Detector: "
                f"{detector_result.target_defect_probability:.2f}"
            )

            critic_result = (
                critic.analyze(
                    method_name=method.name,
                    candidate_context=(
                        candidate_context
                    ),
                    failing_test_context=(
                        failing_test_context
                    ),
                    detector_result=(
                        detector_result
                    ),
                )
            )

            print(
                "  Critic: "
                f"{critic_result.target_defect_probability:.2f}"
            )

            judge_result = (
                judge.analyze(
                    method_name=method.name,
                    candidate_context=(
                        candidate_context
                    ),
                    failing_test_context=(
                        failing_test_context
                    ),
                    detector_result=(
                        detector_result
                    ),
                    critic_result=(
                        critic_result
                    ),
                )
            )

            print(
                "  Judge: "
                f"{judge_result.target_defect_probability:.2f}"
            )

            results.append(
                {
                    "method": method.name,
                    "start_line": (
                        method.start_line
                    ),
                    "end_line": (
                        method.end_line
                    ),
                    "b4_rank": index,
                    "b4_score": (
                        candidate
                        .suspicion_score
                    ),
                    "detector": {
                        "hypothesis": (
                            detector_result
                            .hypothesis
                        ),
                        "supporting_evidence": (
                            detector_result
                            .supporting_evidence
                        ),
                        "target_defect_probability": (
                            detector_result
                            .target_defect_probability
                        ),
                    },
                    "critic": {
                        "agrees_with_detector": (
                            critic_result
                            .agrees_with_detector
                        ),
                        "weaknesses": (
                            critic_result
                            .weaknesses
                        ),
                        "alternative_explanation": (
                            critic_result
                            .alternative_explanation
                        ),
                        "target_defect_probability": (
                            critic_result
                            .target_defect_probability
                        ),
                    },
                    "judge": {
                        "is_target_defect": (
                            judge_result
                            .is_target_defect
                        ),
                        "target_defect_probability": (
                            judge_result
                            .target_defect_probability
                        ),
                        "defect_type": (
                            judge_result
                            .defect_type
                        ),
                        "reason": (
                            judge_result
                            .reason
                        ),
                    },
                }
            )

        results.sort(
            key=lambda item: (
                item["judge"][
                    "target_defect_probability"
                ]
            ),
            reverse=True,
        )

        return results

    # =========================================================
    # Multi-Agent GT metric
    # =========================================================

    @staticmethod
    def find_multi_agent_gt_rank(
        results: list[dict],
        ground_truth_methods: list[JavaMethod],
    ) -> int | None:

        gt_keys = {
            (
                method.name,
                method.start_line,
                method.end_line,
            )
            for method in ground_truth_methods
        }

        for rank, result in enumerate(
            results,
            start=1,
        ):

            key = (
                result["method"],
                result["start_line"],
                result["end_line"],
            )

            if key in gt_keys:
                return rank

        return None

    # =========================================================
    # Main bug experiment
    # =========================================================

    def run(
        self,
    ) -> dict:

        print()
        print("=" * 100)
        print(
            f"CAMD Experiment: "
            f"{self.project}-{self.bug_id}"
        )
        print("=" * 100)

        self.prepare_checkouts()

        modified_classes = (
            self.get_modified_classes()
        )

        print(
            "\nModified classes:"
        )

        for class_name in modified_classes:
            print(
                f"  {class_name}"
            )

        all_methods = []
        all_ground_truth = []

        for class_name in modified_classes:

            buggy_file = (
                self.class_to_source_file(
                    checkout_dir=(
                        self.buggy_dir
                    ),
                    class_name=class_name,
                )
            )

            fixed_file = (
                self.class_to_source_file(
                    checkout_dir=(
                        self.fixed_dir
                    ),
                    class_name=class_name,
                )
            )

            if (
                not buggy_file.exists()
                or not fixed_file.exists()
            ):
                print(
                    f"Skipping missing source: "
                    f"{class_name}"
                )
                continue

            methods = (
                extract_java_methods(
                    buggy_file
                )
            )

            ground_truth = (
                self.get_ground_truth_methods(
                    buggy_file=buggy_file,
                    fixed_file=fixed_file,
                    buggy_methods=methods,
                )
            )

            all_methods.extend(
                methods
            )

            all_ground_truth.extend(
                ground_truth
            )

        if not all_methods:

            raise RuntimeError(
                "No methods were extracted "
                "from modified classes."
            )

        print(
            f"\nTotal candidate methods: "
            f"{len(all_methods)}"
        )

        print(
            "Ground-truth methods:"
        )

        for method in all_ground_truth:

            print(
                f"  {method.name} "
                f"({method.start_line}-"
                f"{method.end_line})"
            )

        # ---------------------------------------------
        # B1
        # ---------------------------------------------

        b1_results = self.run_b1(
            all_methods
        )

        b1_rank = (
            self.find_best_ground_truth_rank(
                results=b1_results,
                ground_truth_methods=(
                    all_ground_truth
                ),
            )
        )

        b1_metrics = (
            self.build_metrics(
                b1_rank
            )
        )

        b1_records = []

        for rank, result in enumerate(
            b1_results,
            start=1,
        ):

            b1_records.append(
                {
                    "rank": rank,
                    "method": (
                        result.method_name
                    ),
                    "start_line": (
                        result.start_line
                    ),
                    "end_line": (
                        result.end_line
                    ),
                    "is_suspicious": (
                        result.is_suspicious
                    ),
                    "suspicion_score": (
                        result.suspicion_score
                    ),
                    "defect_type": (
                        result.defect_type
                    ),
                    "reason": (
                        result.reason
                    ),
                }
            )

        self.write_jsonl(
            self.result_dir
            / "b1_method_ranking.jsonl",
            b1_records,
        )

        # ---------------------------------------------
        # B4
        # ---------------------------------------------

        b4_results = self.run_b4(
            all_methods
        )

        b4_rank = (
            self.find_best_ground_truth_rank(
                results=b4_results,
                ground_truth_methods=(
                    all_ground_truth
                ),
            )
        )

        b4_metrics = (
            self.build_metrics(
                b4_rank
            )
        )

        b4_records = []

        for rank, result in enumerate(
            b4_results,
            start=1,
        ):

            b4_records.append(
                {
                    "rank": rank,
                    "method": (
                        result.method_name
                    ),
                    "start_line": (
                        result.start_line
                    ),
                    "end_line": (
                        result.end_line
                    ),
                    "is_suspicious": (
                        result.is_suspicious
                    ),
                    "suspicion_score": (
                        result.suspicion_score
                    ),
                    "defect_type": (
                        result.defect_type
                    ),
                    "reason": (
                        result.reason
                    ),
                    "selected_context_methods": (
                        result
                        .selected_context_methods
                    ),
                }
            )

        self.write_jsonl(
            self.result_dir
            / "b4_static_ranking.jsonl",
            b4_records,
        )

        # ---------------------------------------------
        # Failing tests
        # ---------------------------------------------

        (
            failing_test_context,
            failing_tests,
        ) = self.build_failing_test_context()

        print(
            "\nFailing tests:"
        )

        for test in failing_tests:
            print(
                f"  {test}"
            )

        # ---------------------------------------------
        # CAMD
        # ---------------------------------------------

        multi_agent_results = (
            self.run_multi_agent(
                methods=all_methods,
                b4_results=b4_results,
                failing_test_context=(
                    failing_test_context
                ),
            )
        )

        self.write_jsonl(
            self.result_dir
            / "multi_agent.jsonl",
            multi_agent_results,
        )

        camd_rank = (
            self.find_multi_agent_gt_rank(
                results=(
                    multi_agent_results
                ),
                ground_truth_methods=(
                    all_ground_truth
                ),
            )
        )

        camd_metrics = (
            self.build_metrics(
                camd_rank
            )
        )

        # ---------------------------------------------
        # Summary
        # ---------------------------------------------

        summary = {
            "project": self.project,
            "bug_id": self.bug_id,

            "modified_classes": (
                modified_classes
            ),

            "failing_tests": (
                failing_tests
            ),

            "ground_truth_methods": [
                {
                    "name": method.name,
                    "start_line": (
                        method.start_line
                    ),
                    "end_line": (
                        method.end_line
                    ),
                }
                for method
                in all_ground_truth
            ],

            "candidate_method_count": (
                len(all_methods)
            ),

            "b1_method_only": (
                b1_metrics
            ),

            "b4_static_aware": (
                b4_metrics
            ),

            "camd_multi_agent": (
                camd_metrics
            ),
        }

        self.write_json(
            self.result_dir
            / "summary.json",
            summary,
        )

        print()
        print("=" * 100)
        print(
            f"{self.project}-{self.bug_id} "
            "Summary"
        )
        print("=" * 100)

        print(
            f"B1 rank: "
            f"{b1_metrics['rank']} "
            f"RR={b1_metrics['rr']:.4f}"
        )

        print(
            f"B4 rank: "
            f"{b4_metrics['rank']} "
            f"RR={b4_metrics['rr']:.4f}"
        )

        print(
            f"CAMD rank: "
            f"{camd_metrics['rank']} "
            f"RR={camd_metrics['rr']:.4f}"
        )

        return summary
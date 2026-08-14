import json
from dataclasses import asdict
from pathlib import Path

from camd.agents.critic_agent import CriticAgent
from camd.agents.detector_agent import DetectorAgent
from camd.agents.judge_agent import JudgeAgent
from camd.context.method_extractor import extract_java_methods
from camd.evaluation.failing_test_extractor import FailingTestExtractor
from camd.evaluation.test_context_builder import TestContextBuilder
from camd.llm.client import OpenAIClient
from camd.static.ast_analyzer import JavaASTAnalyzer
from camd.static.evidence_builder import StaticEvidenceBuilder


class AdaptiveCandidateRunner:

    def __init__(
        self,
        project_root: Path,
        threshold: float = 0.5,
        initial_top_k: int = 5,
        expanded_top_k: int = 10,
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

        self.threshold = threshold
        self.initial_top_k = initial_top_k
        self.expanded_top_k = expanded_top_k

        self.llm_client = OpenAIClient()

        self.detector_agent = DetectorAgent(
            llm_client=self.llm_client
        )

        self.critic_agent = CriticAgent(
            llm_client=self.llm_client
        )

        self.judge_agent = JudgeAgent(
            llm_client=self.llm_client
        )

        self.ast_analyzer = JavaASTAnalyzer()
        self.evidence_builder = StaticEvidenceBuilder()
        self.test_context_builder = TestContextBuilder()

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
    def _load_jsonl(
        path: Path,
    ) -> list[dict]:

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

        return records

    @staticmethod
    def _save_jsonl(
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
                    + "\n"
                )

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
    def _judge_score(
        record: dict,
    ) -> float:

        judge = record.get(
            "judge"
        )

        if not isinstance(
            judge,
            dict,
        ):
            return 0.0

        return float(
            judge.get(
                "target_defect_probability",
                0.0,
            )
        )

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

            candidate = root / relative

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

    def _load_b4_ranking(
        self,
        project: str,
        bug_id: int,
    ) -> list[dict]:

        path = (
            self.results_root
            / f"{project}_{bug_id}"
            / "b4_static_ranking.jsonl"
        )

        if not path.exists():
            raise RuntimeError(
                f"Missing B4 ranking: {path}"
            )

        records = self._load_jsonl(
            path
        )

        def score(
            item: dict,
        ) -> float:

            return float(
                item.get(
                    "suspicion_score",
                    item.get(
                        "score",
                        0.0,
                    ),
                )
            )

        return sorted(
            records,
            key=score,
            reverse=True,
        )

    def _load_existing_multi_agent(
        self,
        project: str,
        bug_id: int,
    ) -> tuple[
        list[dict],
        Path,
    ]:

        bug_dir = (
            self.results_root
            / f"{project}_{bug_id}"
        )

        candidates = [
            bug_dir / "multi_agent_adaptive.jsonl",
            bug_dir / "multi_agent_expanded_test.jsonl",
            bug_dir / "multi_agent.jsonl",
        ]

        for path in candidates:

            if path.exists():

                records = (
                    self._load_jsonl(
                        path
                    )
                )

                if records:
                    return records, path

        raise RuntimeError(
            "No saved Multi-Agent results found."
        )

    def _resolve_method(
        self,
        checkout_dir: Path,
        modified_classes: list[str],
        ranking_record: dict,
    ):

        method_name = (
            ranking_record.get(
                "method"
            )
            or ranking_record.get(
                "method_name"
            )
        )

        start_line = (
            ranking_record.get(
                "start_line"
            )
        )

        end_line = (
            ranking_record.get(
                "end_line"
            )
        )

        matches = []

        for class_name in modified_classes:

            source_file = (
                self._find_source_file(
                    checkout_dir,
                    class_name,
                )
            )

            if source_file is None:
                continue

            methods = (
                extract_java_methods(
                    source_file
                )
            )

            for method in methods:

                if method.name != method_name:
                    continue

                if (
                    start_line is not None
                    and end_line is not None
                ):

                    if (
                        method.start_line
                        != start_line
                        or method.end_line
                        != end_line
                    ):
                        continue

                matches.append(
                    (
                        class_name,
                        source_file,
                        method,
                    )
                )

        if len(matches) != 1:

            raise RuntimeError(
                "Could not uniquely resolve "
                f"candidate method: "
                f"{method_name} "
                f"{start_line}-{end_line}"
            )

        return matches[0]

    def _run_agents_for_candidate(
        self,
        candidate: dict,
        checkout_dir: Path,
        modified_classes: list[str],
        failing_test_context: str,
    ) -> dict:

        (
            class_name,
            source_file,
            method,
        ) = self._resolve_method(
            checkout_dir=checkout_dir,
            modified_classes=modified_classes,
            ranking_record=candidate,
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

        candidate_context = f"""
    SOURCE CODE
    ===========

    {method.code}


    STATIC EVIDENCE
    ===============

    {static_text}
    """.strip()

        detector = (
            self.detector_agent.analyze(
                method_name=method.name,
                candidate_context=(
                    candidate_context
                ),
                failing_test_context=(
                    failing_test_context
                ),
            )
        )

        critic = (
            self.critic_agent.analyze(
                method_name=method.name,
                candidate_context=(
                    candidate_context
                ),
                failing_test_context=(
                    failing_test_context
                ),
                detector_result=(
                    detector
                ),
            )
        )

        judge = (
            self.judge_agent.analyze(
                method_name=method.name,
                candidate_context=(
                    candidate_context
                ),
                failing_test_context=(
                    failing_test_context
                ),
                detector_result=(
                    detector
                ),
                critic_result=(
                    critic
                ),
            )
        )

        return {
            "method": (
                method.name
            ),

            "start_line": (
                method.start_line
            ),

            "end_line": (
                method.end_line
            ),

            "class_name": (
                class_name
            ),

            "b4_rank": (
                candidate.get(
                    "rank"
                )
                or candidate.get(
                    "b4_rank"
                )
            ),

            "b4_score": (
                candidate.get(
                    "suspicion_score",
                    candidate.get(
                        "b4_score",
                    ),
                )
            ),

            "detector": (
                asdict(
                    detector
                )
            ),

            "critic": (
                asdict(
                    critic
                )
            ),

            "judge": (
                asdict(
                    judge
                )
            ),
        }

    def run_bug(
        self,
        bug_record: dict,
    ) -> dict:

        project = (
            bug_record["project"]
        )

        bug_id = (
            bug_record["bug_id"]
        )

        buggy_dir = (
            self.checkouts_root
            / f"{project}_{bug_id}b"
        )

        existing_records, existing_path = (
            self._load_existing_multi_agent(
                project=project,
                bug_id=bug_id,
            )
        )

        existing_ranked = sorted(
            existing_records,
            key=self._judge_score,
            reverse=True,
        )

        initial_best_score = (
            self._judge_score(
                existing_ranked[0]
            )
        )

        initial_best_method = (
            existing_ranked[0].get(
                "method"
            )
        )

        triggered = (
            initial_best_score
            < self.threshold
        )

        if not triggered:

            result = {
                "project": project,
                "bug_id": bug_id,
                "triggered": False,
                "threshold": self.threshold,
                "initial_top_k": (
                    self.initial_top_k
                ),
                "expanded_top_k": (
                    self.expanded_top_k
                ),
                "initial_best_method": (
                    initial_best_method
                ),
                "initial_best_score": (
                    initial_best_score
                ),
                "final_best_method": (
                    initial_best_method
                ),
                "final_best_score": (
                    initial_best_score
                ),
                "new_candidates_evaluated": 0,
                "existing_source": (
                    existing_path.name
                ),
            }

            return result

        b4_ranking = (
            self._load_b4_ranking(
                project=project,
                bug_id=bug_id,
            )
        )

        if (
            len(b4_ranking)
            <= self.initial_top_k
        ):

            return {
                "project": project,
                "bug_id": bug_id,
                "triggered": True,
                "expanded": False,
                "reason": (
                    "No additional B4 candidates "
                    "available."
                ),
                "initial_best_method": (
                    initial_best_method
                ),
                "initial_best_score": (
                    initial_best_score
                ),
            }

        new_candidates = (
            b4_ranking[
                self.initial_top_k:
                min(
                    self.expanded_top_k,
                    len(b4_ranking),
                )
            ]
        )

        failing_test_context = (
            self._build_failing_test_context(
                buggy_dir
            )
        )

        modified_classes = (
            bug_record.get(
                "modified_classes",
                [],
            )
        )

        new_records = []

        for candidate in new_candidates:

            new_record = (
                self._run_agents_for_candidate(
                    candidate=candidate,
                    checkout_dir=buggy_dir,
                    modified_classes=(
                        modified_classes
                    ),
                    failing_test_context=(
                        failing_test_context
                    ),
                )
            )

            new_records.append(
                new_record
            )

        combined = (
            existing_records
            + new_records
        )

        combined_ranked = sorted(
            combined,
            key=self._judge_score,
            reverse=True,
        )

        final_best = (
            combined_ranked[0]
        )

        output_jsonl = (
            self.results_root
            / f"{project}_{bug_id}"
            / "multi_agent_adaptive.jsonl"
        )

        self._save_jsonl(
            output_jsonl,
            combined_ranked,
        )

        summary = {
            "project": project,
            "bug_id": bug_id,

            "triggered": True,
            "expanded": True,

            "threshold": (
                self.threshold
            ),

            "initial_top_k": (
                self.initial_top_k
            ),

            "expanded_top_k": (
                self.expanded_top_k
            ),

            "existing_source": (
                existing_path.name
            ),

            "initial_best_method": (
                initial_best_method
            ),

            "initial_best_score": (
                initial_best_score
            ),

            "new_candidates_evaluated": (
                len(new_records)
            ),

            "final_best_method": (
                final_best.get(
                    "method"
                )
            ),

            "final_best_score": (
                self._judge_score(
                    final_best
                )
            ),

            "final_ranking": [
                {
                    "rank": rank,
                    "method": (
                        record.get(
                            "method"
                        )
                    ),
                    "start_line": (
                        record.get(
                            "start_line"
                        )
                    ),
                    "end_line": (
                        record.get(
                            "end_line"
                        )
                    ),
                    "judge_score": (
                        self._judge_score(
                            record
                        )
                    ),
                    "b4_rank": (
                        record.get(
                            "b4_rank"
                        )
                    ),
                }
                for rank, record
                in enumerate(
                    combined_ranked,
                    start=1,
                )
            ],
        }

        output_summary = (
            self.results_root
            / f"{project}_{bug_id}"
            / "adaptive_candidate_summary.json"
        )

        self._save_json(
            output_summary,
            summary,
        )

        return summary
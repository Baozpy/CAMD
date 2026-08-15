import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from camd.context.method_extractor import (
    extract_java_methods,
)
from camd.evaluation.diff_ground_truth import (
    extract_changed_ranges,
)
from camd.evaluation.failing_test_extractor import (
    FailingTestExtractor,
)
from camd.evaluation.test_context_builder import (
    TestContextBuilder,
)
from camd.static.ast_analyzer import (
    JavaASTAnalyzer,
)
from camd.static.evidence_builder import (
    StaticEvidenceBuilder,
)


@dataclass
class QLoRASample:
    project: str
    bug_id: int

    class_name: str

    method_name: str
    start_line: int
    end_line: int

    method_length: int

    label: int
    is_target_defect: bool

    input: str
    output: dict


class QLoRADatasetBuilder:

    def __init__(
        self,
        project_root: Path,
        max_negatives_per_positive: int | None = 8,
    ):
        self.project_root = project_root

        self.checkouts_root = (
            project_root
            / "data"
            / "defects4j"
            / "checkouts"
        )

        self.defects4j_bin = (
            project_root
            / "external"
            / "defects4j"
            / "framework"
            / "bin"
            / "defects4j"
        )

        self.max_negatives_per_positive = (
            max_negatives_per_positive
        )

        self.ast_analyzer = (
            JavaASTAnalyzer()
        )

        self.evidence_builder = (
            StaticEvidenceBuilder()
        )

        self.test_context_builder = (
            TestContextBuilder()
        )

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
        )

        if result.returncode != 0:

            raise RuntimeError(
                "Command failed:\n"
                + " ".join(command)
                + "\n\nSTDOUT:\n"
                + result.stdout
                + "\n\nSTDERR:\n"
                + result.stderr
            )

        return result.stdout.strip()

    def _checkout_version(
        self,
        project: str,
        bug_id: int,
        version: str,
    ) -> Path:

        checkout_dir = (
            self.checkouts_root
            / f"{project}_{bug_id}{version}"
        )

        if (
            checkout_dir.exists()
            and any(
                checkout_dir.iterdir()
            )
        ):
            return checkout_dir

        checkout_dir.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._run_command(
            [
                str(
                    self.defects4j_bin
                ),
                "checkout",
                "-p",
                project,
                "-v",
                f"{bug_id}{version}",
                "-w",
                str(
                    checkout_dir
                ),
            ]
        )

        return checkout_dir

    def ensure_checkouts(
        self,
        project: str,
        bug_id: int,
    ) -> tuple[
        Path,
        Path,
    ]:

        buggy_dir = (
            self._checkout_version(
                project=project,
                bug_id=bug_id,
                version="b",
            )
        )

        fixed_dir = (
            self._checkout_version(
                project=project,
                bug_id=bug_id,
                version="f",
            )
        )

        return (
            buggy_dir,
            fixed_dir,
        )

    def _export_property(
        self,
        checkout_dir: Path,
        property_name: str,
    ) -> str:

        return self._run_command(
            [
                str(
                    self.defects4j_bin
                ),
                "export",
                "-p",
                property_name,
                "-w",
                str(
                    checkout_dir
                ),
            ]
        )

    def get_modified_classes(
        self,
        checkout_dir: Path,
    ) -> list[str]:

        output = (
            self._export_property(
                checkout_dir=checkout_dir,
                property_name=(
                    "classes.modified"
                ),
            )
        )

        classes = []

        for line in (
            output.splitlines()
        ):

            value = (
                line.strip()
            )

            if value:
                classes.append(
                    value
                )

        return classes

    def _find_source_file(
        self,
        checkout_dir: Path,
        class_name: str,
    ) -> Path | None:

        relative_path = Path(
            *class_name.split(".")
        ).with_suffix(".java")

        candidate_roots = [
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

        for root in candidate_roots:

            candidate = (
                root
                / relative_path
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

        filtered = []

        for path in matches:

            lower_parts = {
                part.lower()
                for part
                in path.parts
            }

            if (
                "test" in lower_parts
                or "tests" in lower_parts
            ):
                continue

            filtered.append(
                path
            )

        if len(filtered) == 1:
            return filtered[0]

        return None

    def _build_failing_test_context(
        self,
        buggy_dir: Path,
    ) -> str:

        extractor = (
            FailingTestExtractor(
                checkout_dir=(
                    buggy_dir
                )
            )
        )

        tests = (
            extractor.extract()
        )

        parts = []

        for test in tests:

            parts.append(
                "Failing test: "
                f"{test.full_name}"
            )

            parts.append("")

            expanded = False

            if (
                test.source_file
                is not None
                and test.method_name
            ):

                try:

                    context = (
                        self.test_context_builder
                        .build(
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
    def _method_has_changed_line(
        method,
        changed_ranges,
    ) -> bool:

        for changed in changed_ranges:

            overlaps = not (
                method.end_line
                < changed.start_line
                or method.start_line
                > changed.end_line
            )

            if overlaps:
                return True

        return False

    def _build_static_text(
        self,
        method,
    ) -> str:

        evidence = (
            self.ast_analyzer.analyze(
                method
            )
        )

        return (
            self.evidence_builder
            .build_text(
                evidence
            )
        )

    @staticmethod
    def _build_model_input(
        project: str,
        bug_id: int,
        class_name: str,
        method,
        static_text: str,
        failing_test_context: str,
    ) -> str:

        return f"""
You are given one candidate Java method from a buggy program.

Your task is to determine whether this method is a target defect
responsible for the CURRENT failing test.

Do not search for unrelated defects.
Do not propose a patch.

PROJECT
=======

{project}-{bug_id}


CANDIDATE CLASS
===============

{class_name}


CANDIDATE METHOD
================

Method:
{method.name}

Lines:
{method.start_line}-{method.end_line}

Source:
{method.code}


STATIC EVIDENCE
===============

{static_text}


FAILING TEST CONTEXT
====================

{failing_test_context}


Return whether this candidate method is a target defect for the
current failing test.
""".strip()

    def _build_sample(
        self,
        project: str,
        bug_id: int,
        class_name: str,
        method,
        is_target: bool,
        failing_test_context: str,
    ) -> QLoRASample:

        static_text = (
            self._build_static_text(
                method
            )
        )

        model_input = (
            self._build_model_input(
                project=project,
                bug_id=bug_id,
                class_name=class_name,
                method=method,
                static_text=static_text,
                failing_test_context=(
                    failing_test_context
                ),
            )
        )

        return QLoRASample(
            project=project,
            bug_id=bug_id,

            class_name=class_name,

            method_name=(
                method.name
            ),

            start_line=(
                method.start_line
            ),

            end_line=(
                method.end_line
            ),

            method_length=(
                method.end_line
                - method.start_line
                + 1
            ),

            label=(
                1
                if is_target
                else 0
            ),

            is_target_defect=(
                is_target
            ),

            input=(
                model_input
            ),

            output={
                "is_target_defect": (
                    is_target
                )
            },
        )

    def _select_hard_negatives(
        self,
        positives: list[QLoRASample],
        negatives: list[QLoRASample],
    ) -> list[QLoRASample]:

        if (
            self.max_negatives_per_positive
            is None
        ):
            return negatives

        if not positives:
            return []

        max_negative_count = (
            self.max_negatives_per_positive
            * len(
                positives
            )
        )

        if (
            len(negatives)
            <= max_negative_count
        ):
            return negatives

        positive_lengths = [
            sample.method_length
            for sample
            in positives
        ]

        positive_names = {
            sample.method_name
            for sample
            in positives
        }

        positive_classes = {
            sample.class_name
            for sample
            in positives
        }

        def negative_priority(
            sample: QLoRASample,
        ) -> tuple:

            same_class = (
                0
                if sample.class_name
                in positive_classes
                else 1
            )

            same_name = (
                0
                if sample.method_name
                in positive_names
                else 1
            )

            length_distance = min(
                abs(
                    sample.method_length
                    - positive_length
                )
                for positive_length
                in positive_lengths
            )

            return (
                same_class,
                same_name,
                length_distance,
                sample.class_name,
                sample.method_name,
                sample.start_line,
            )

        ranked = sorted(
            negatives,
            key=negative_priority,
        )

        return ranked[
            :max_negative_count
        ]

    def build_bug_samples(
        self,
        project: str,
        bug_id: int,
    ) -> list[QLoRASample]:

        (
            buggy_dir,
            fixed_dir,
        ) = self.ensure_checkouts(
            project=project,
            bug_id=bug_id,
        )

        modified_classes = (
            self.get_modified_classes(
                buggy_dir
            )
        )

        if not modified_classes:

            raise RuntimeError(
                "No modified classes "
                f"found for "
                f"{project}-{bug_id}"
            )

        failing_test_context = (
            self._build_failing_test_context(
                buggy_dir
            )
        )

        positives = []
        negatives = []

        for class_name in (
            modified_classes
        ):

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

                print(
                    "WARNING: "
                    "source file could not "
                    "be resolved for "
                    f"{class_name}"
                )

                continue

            methods = (
                extract_java_methods(
                    buggy_file
                )
            )

            changed_ranges = (
                extract_changed_ranges(
                    buggy_file=(
                        buggy_file
                    ),
                    fixed_file=(
                        fixed_file
                    ),
                )
            )

            for method in methods:

                is_target = (
                    self._method_has_changed_line(
                        method=method,
                        changed_ranges=(
                            changed_ranges
                        ),
                    )
                )

                sample = (
                    self._build_sample(
                        project=project,
                        bug_id=bug_id,
                        class_name=(
                            class_name
                        ),
                        method=method,
                        is_target=(
                            is_target
                        ),
                        failing_test_context=(
                            failing_test_context
                        ),
                    )
                )

                if is_target:
                    positives.append(
                        sample
                    )

                else:
                    negatives.append(
                        sample
                    )

        if not positives:

            raise RuntimeError(
                "Dataset builder produced "
                "zero positive samples for "
                f"{project}-{bug_id}. "
                "Check diff-to-method "
                "ground-truth mapping."
            )

        selected_negatives = (
            self._select_hard_negatives(
                positives=positives,
                negatives=negatives,
            )
        )

        samples = (
            positives
            + selected_negatives
        )

        samples.sort(
            key=lambda item: (
                item.class_name,
                item.start_line,
                item.method_name,
            )
        )

        return samples

    @staticmethod
    def save_samples(
        samples: list[QLoRASample],
        output_file: Path,
    ) -> None:

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_file.open(
            "w",
            encoding="utf-8",
        ) as file:

            for sample in samples:

                file.write(
                    json.dumps(
                        asdict(
                            sample
                        ),
                        ensure_ascii=False,
                    )
                )

                file.write(
                    "\n"
                )
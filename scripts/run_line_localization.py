from pathlib import Path

from camd.context.method_extractor import (
    extract_java_methods,
)

from camd.detectors.line_ranker import (
    LineRanker,
)

from camd.evaluation.diff_ground_truth import (
    extract_changed_ranges,
)

from camd.evaluation.failing_test_extractor import (
    FailingTestExtractor,
)

from camd.evaluation.line_evaluator import (
    evaluate_line_ranking,
)

from camd.evaluation.test_context_builder import (
    TestContextBuilder,
)

from camd.llm.client import (
    OpenAIClient,
)

from camd.static.ast_analyzer import (
    JavaASTAnalyzer,
)

from camd.static.evidence_builder import (
    StaticEvidenceBuilder,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

BUGGY_DIR = (
    PROJECT_ROOT
    / "data"
    / "defects4j"
    / "checkouts"
    / "Lang_1b"
)

FIXED_DIR = (
    PROJECT_ROOT
    / "data"
    / "defects4j"
    / "checkouts"
    / "Lang_1f"
)

BUGGY_FILE = (
    BUGGY_DIR
    / "src"
    / "main"
    / "java"
    / "org"
    / "apache"
    / "commons"
    / "lang3"
    / "math"
    / "NumberUtils.java"
)

FIXED_FILE = (
    FIXED_DIR
    / "src"
    / "main"
    / "java"
    / "org"
    / "apache"
    / "commons"
    / "lang3"
    / "math"
    / "NumberUtils.java"
)

TARGET_METHOD = (
    "createNumber"
)

TOP_K = 10


def add_line_numbers(
    code: str,
    start_line: int,
) -> str:

    lines = (
        code.splitlines()
    )

    output = []

    for offset, line in enumerate(
        lines
    ):

        absolute_line = (
            start_line
            + offset
        )

        output.append(
            f"{absolute_line:5d}: "
            f"{line}"
        )

    return "\n".join(
        output
    )


def build_ground_truth_lines():

    changed_ranges = (
        extract_changed_ranges(
            buggy_file=BUGGY_FILE,
            fixed_file=FIXED_FILE,
        )
    )

    lines = set()

    for changed in changed_ranges:

        for line in range(
            changed.start_line,
            changed.end_line + 1,
        ):

            lines.add(
                line
            )

    return sorted(
        lines
    )


def build_failing_test_context():

    failing_extractor = (
        FailingTestExtractor(
            checkout_dir=BUGGY_DIR
        )
    )

    context_builder = (
        TestContextBuilder()
    )

    tests = (
        failing_extractor.extract()
    )

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
                    context_builder.build(
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


def main():

    methods = (
        extract_java_methods(
            BUGGY_FILE
        )
    )

    target_method = next(
        (
            method
            for method in methods
            if (
                method.name
                == TARGET_METHOD
            )
        ),
        None,
    )

    if target_method is None:

        raise RuntimeError(
            f"Target method not found: "
            f"{TARGET_METHOD}"
        )

    ground_truth_lines = (
        build_ground_truth_lines()
    )

    print("=" * 100)
    print(
        "CAMD - Line-Level Localization"
    )
    print("=" * 100)

    print(
        f"Target method: "
        f"{target_method.name}"
    )

    print(
        f"Method lines: "
        f"{target_method.start_line}-"
        f"{target_method.end_line}"
    )

    print(
        f"Ground-truth changed lines: "
        f"{ground_truth_lines}"
    )

    method_code = (
        add_line_numbers(
            code=target_method.code,
            start_line=(
                target_method.start_line
            ),
        )
    )

    ast_analyzer = (
        JavaASTAnalyzer()
    )

    evidence_builder = (
        StaticEvidenceBuilder()
    )

    static_evidence = (
        ast_analyzer.analyze(
            target_method
        )
    )

    static_text = (
        evidence_builder.build_text(
            static_evidence
        )
    )

    failing_test_context = (
        build_failing_test_context()
    )

    client = (
        OpenAIClient()
    )

    ranker = (
        LineRanker(
            llm_client=client
        )
    )

    results = (
        ranker.rank(
            method_name=(
                target_method.name
            ),

            method_code_with_lines=(
                method_code
            ),

            static_evidence=(
                static_text
            ),

            failing_test_context=(
                failing_test_context
            ),

            top_k=TOP_K,
        )
    )

    print()
    print("=" * 100)
    print(
        "Top Suspicious Lines"
    )
    print("=" * 100)

    for rank, result in enumerate(
        results,
        start=1,
    ):

        print(
            f"#{rank:<3} "
            f"line={result.line:<5} "
            f"score={result.score:.2f}"
        )

        print(
            f"     reason="
            f"{result.reason}"
        )

    predicted_lines = [
        result.line
        for result in results
    ]

    metrics = (
        evaluate_line_ranking(
            predicted_lines=(
                predicted_lines
            ),
            ground_truth_lines=(
                ground_truth_lines
            ),
        )
    )

    print()
    print("=" * 100)
    print(
        "Line-Level Evaluation"
    )
    print("=" * 100)

    print(
        f"Ground truth: "
        f"{metrics.ground_truth_lines}"
    )

    print(
        f"Predicted: "
        f"{metrics.predicted_lines}"
    )

    print(
        f"First hit rank: "
        f"{metrics.first_hit_rank}"
    )

    print(
        f"MRR: "
        f"{metrics.reciprocal_rank:.4f}"
    )

    print(
        f"Top-1: "
        f"{metrics.top_1}"
    )

    print(
        f"Top-3: "
        f"{metrics.top_3}"
    )

    print(
        f"Top-5: "
        f"{metrics.top_5}"
    )

    print(
        f"Top-10: "
        f"{metrics.top_10}"
    )


if __name__ == "__main__":
    main()
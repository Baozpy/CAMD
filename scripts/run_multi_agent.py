import json
from pathlib import Path

from camd.agents.critic_agent import (
    CriticAgent,
)
from camd.agents.detector_agent import (
    DetectorAgent,
)
from camd.agents.judge_agent import (
    JudgeAgent,
)
from camd.context.method_extractor import (
    extract_java_methods,
)
from camd.context.semantic_context_builder import (
    SemanticContextBuilder,
)
from camd.detectors.static_context_ranker import (
    format_semantic_context,
)
from camd.evaluation.failing_test_extractor import (
    FailingTestExtractor,
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

CHECKOUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "defects4j"
    / "checkouts"
    / "Lang_1b"
)

SOURCE_FILE = (
    CHECKOUT_DIR
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

INPUT_RANKING_FILE = (
    PROJECT_ROOT
    / "results"
    / "static_context_ranking_Lang_1.jsonl"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "multi_agent_Lang_1.jsonl"
)

GROUND_TRUTH_METHOD = (
    "createNumber"
)

TOP_K = 5


def load_top_candidates():

    candidates = []

    with INPUT_RANKING_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            if not line.strip():
                continue

            candidates.append(
                json.loads(
                    line
                )
            )

    candidates.sort(
        key=lambda item: (
            item["rank"]
        )
    )

    return candidates[:TOP_K]


def build_failing_test_context():

    extractor = (
        FailingTestExtractor(
            checkout_dir=CHECKOUT_DIR
        )
    )

    tests = extractor.extract()

    if not tests:

        raise RuntimeError(
            "No failing tests found."
        )

    output = []

    for test in tests:

        output.append(
            f"Test: {test.full_name}"
        )

        if test.source_file:

            output.append(
                f"Source file: "
                f"{test.source_file}"
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

    return "\n".join(
        output
    )


def find_method(
    methods,
    candidate,
):

    for method in methods:

        if (
            method.name
            == candidate["method"]
            and method.start_line
            == candidate["start_line"]
            and method.end_line
            == candidate["end_line"]
        ):

            return method

    return None


def build_candidate_context(
    method,
    semantic_builder,
    ast_analyzer,
    evidence_builder,
):

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


def reciprocal_rank(
    rank,
):

    if rank is None:
        return 0.0

    return 1.0 / rank


def save_results(
    results,
):

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        for rank, item in enumerate(
            results,
            start=1,
        ):

            record = {
                "rank": rank,
                "method": (
                    item["method"]
                ),
                "start_line": (
                    item["start_line"]
                ),
                "end_line": (
                    item["end_line"]
                ),
                "b4_rank": (
                    item["b4_rank"]
                ),
                "b4_score": (
                    item["b4_score"]
                ),
                "detector": (
                    item["detector"]
                ),
                "critic": (
                    item["critic"]
                ),
                "judge": (
                    item["judge"]
                ),
            }

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
            )

            file.write("\n")


def print_results(
    results,
):

    print()
    print("=" * 100)
    print(
        "CAMD Multi-Agent Final Ranking"
    )
    print("=" * 100)

    for rank, item in enumerate(
        results,
        start=1,
    ):

        judge = item["judge"]

        print(
            f"#{rank:<3} "
            f"{item['method']:<25} "
            f"probability="
            f"{judge['target_defect_probability']:.2f}"
        )

        print(
            f"     B4 rank="
            f"{item['b4_rank']} "
            f"B4 score="
            f"{item['b4_score']:.2f}"
        )

        print(
            f"     target="
            f"{judge['is_target_defect']}"
        )

        print(
            f"     type="
            f"{judge['defect_type']}"
        )

        print(
            f"     reason="
            f"{judge['reason']}"
        )

        print()


def print_evaluation(
    results,
):

    gt_rank = None

    for rank, item in enumerate(
        results,
        start=1,
    ):

        if (
            item["method"]
            == GROUND_TRUTH_METHOD
        ):
            gt_rank = rank
            break

    print()
    print("=" * 100)
    print(
        "CAMD Multi-Agent Localization Evaluation"
    )
    print("=" * 100)

    print(
        f"Ground-truth method: "
        f"{GROUND_TRUTH_METHOD}"
    )

    if gt_rank is None:

        print(
            "Ground-truth method "
            "not found in candidate set."
        )

        print(
            "Reciprocal rank: 0.0000"
        )

        return

    print(
        f"Ground-truth rank: "
        f"{gt_rank}"
    )

    print(
        f"Reciprocal rank: "
        f"{reciprocal_rank(gt_rank):.4f}"
    )

    print(
        f"Top-1 localization: "
        f"{gt_rank <= 1}"
    )

    print(
        f"Top-3 localization: "
        f"{gt_rank <= 3}"
    )

    print(
        f"Top-5 localization: "
        f"{gt_rank <= 5}"
    )

    print("=" * 100)


def main():

    print("=" * 100)
    print(
        "CAMD - Test-Aware "
        "Multi-Agent Defect Verification"
    )
    print("=" * 100)

    candidates = (
        load_top_candidates()
    )

    print(
        f"\nLoaded Top-{len(candidates)} "
        f"candidates from B4."
    )

    methods = (
        extract_java_methods(
            SOURCE_FILE
        )
    )

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

    failing_test_context = (
        build_failing_test_context()
    )

    client = OpenAIClient()

    detector = DetectorAgent(
        llm_client=client
    )

    critic = CriticAgent(
        llm_client=client
    )

    judge = JudgeAgent(
        llm_client=client
    )

    results = []

    for index, candidate in enumerate(
        candidates,
        start=1,
    ):

        method = find_method(
            methods=methods,
            candidate=candidate,
        )

        if method is None:

            print(
                f"[{index}/{len(candidates)}] "
                f"Method not found: "
                f"{candidate['method']}"
            )

            continue

        print()
        print(
            f"[{index}/{len(candidates)}] "
            f"Candidate: "
            f"{method.name} "
            f"({method.start_line}-"
            f"{method.end_line})"
        )

        candidate_context = (
            build_candidate_context(
                method=method,
                semantic_builder=(
                    semantic_builder
                ),
                ast_analyzer=(
                    ast_analyzer
                ),
                evidence_builder=(
                    evidence_builder
                ),
            )
        )

        print(
            "  Running Detector..."
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
            "  Detector probability: "
            f"{detector_result.target_defect_probability:.2f}"
        )

        print(
            "  Running Critic..."
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
            "  Critic probability: "
            f"{critic_result.target_defect_probability:.2f}"
        )

        print(
            "  Running Judge..."
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
            "  Judge probability: "
            f"{judge_result.target_defect_probability:.2f}"
        )

        results.append(
            {
                "method": (
                    method.name
                ),
                "start_line": (
                    method.start_line
                ),
                "end_line": (
                    method.end_line
                ),

                "b4_rank": (
                    candidate["rank"]
                ),

                "b4_score": (
                    candidate[
                        "suspicion_score"
                    ]
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

    save_results(
        results
    )

    print_results(
        results
    )

    print_evaluation(
        results
    )

    print(
        f"\nResults saved to:\n"
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
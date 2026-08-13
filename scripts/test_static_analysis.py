from pathlib import Path

from camd.context.method_extractor import (
    extract_java_methods,
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

SOURCE_FILE = (
    PROJECT_ROOT
    / "data"
    / "defects4j"
    / "checkouts"
    / "Lang_1b"
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


def main():

    methods = extract_java_methods(
        SOURCE_FILE
    )

    target = next(
        method
        for method in methods
        if method.name
        == "createNumber"
    )

    analyzer = JavaASTAnalyzer()

    evidence = analyzer.analyze(
        target
    )

    builder = StaticEvidenceBuilder()

    formatted = builder.build_text(
        evidence
    )

    print("=" * 80)
    print(
        "CAMD - Static Analysis Test"
    )
    print("=" * 80)

    print()
    print(
        formatted
    )

    print("=" * 80)


if __name__ == "__main__":
    main()
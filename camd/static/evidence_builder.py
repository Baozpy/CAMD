from camd.static.models import (
    StaticEvidence,
)


class StaticEvidenceBuilder:

    def build_text(
        self,
        evidence: StaticEvidence,
    ) -> str:

        output = []

        output.append(
            "STATIC ANALYSIS EVIDENCE"
        )
        output.append(
            "========================"
        )

        output.append(
            f"Method: "
            f"{evidence.method_name}"
        )

        output.append(
            f"Lines: "
            f"{evidence.start_line}-"
            f"{evidence.end_line}"
        )

        output.append("")

        output.append(
            "Structural summary:"
        )

        output.append(
            f"- Conditional branches: "
            f"{evidence.condition_count}"
        )

        output.append(
            f"- Loops: "
            f"{evidence.loop_count}"
        )

        output.append(
            f"- Return statements: "
            f"{evidence.return_count}"
        )

        output.append(
            f"- Throw statements: "
            f"{evidence.throw_count}"
        )

        output.append("")

        self._append_list(
            output=output,
            title="Method calls",
            values=evidence.method_calls,
        )

        self._append_list(
            output=output,
            title="Comparisons",
            values=evidence.comparisons,
        )

        self._append_list(
            output=output,
            title="Numeric literals",
            values=evidence.numeric_literals,
        )

        self._append_list(
            output=output,
            title="Null checks",
            values=evidence.null_checks,
        )

        self._append_list(
            output=output,
            title="Thrown exceptions",
            values=evidence.thrown_exceptions,
        )

        return "\n".join(
            output
        )

    @staticmethod
    def _append_list(
        output: list[str],
        title: str,
        values: list[str],
    ) -> None:

        output.append(
            f"{title}:"
        )

        if not values:

            output.append(
                "- None"
            )

        else:

            for value in values:
                output.append(
                    f"- {value}"
                )

        output.append("")
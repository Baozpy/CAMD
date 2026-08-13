from camd.context.context_builder import MethodContext
from camd.context.method_extractor import JavaMethod


def add_line_numbers(
    method: JavaMethod,
) -> str:

    lines = method.code.splitlines()

    numbered_lines = []

    for offset, line in enumerate(lines):

        absolute_line = (
            method.start_line + offset
        )

        numbered_lines.append(
            f"{absolute_line:5d}: {line}"
        )

    return "\n".join(numbered_lines)


def extract_method_header(
    method: JavaMethod,
) -> str:

    lines = method.code.splitlines()

    if not lines:
        return ""

    header_lines = []

    for line in lines:

        header_lines.append(
            line.strip()
        )

        if "{" in line:
            break

    return " ".join(
        header_lines
    )


def format_related_methods(
    title: str,
    methods: list[JavaMethod],
) -> str:

    output = [
        title,
        "-" * len(title),
    ]

    if not methods:

        output.append("None")

        return "\n".join(output)

    for method in methods:

        header = extract_method_header(
            method
        )

        output.append(
            f"{method.name} "
            f"({method.start_line}-"
            f"{method.end_line})"
        )

        output.append(
            header
        )

        output.append("")

    return "\n".join(output)


def format_method_context(
    context: MethodContext,
) -> str:

    target_code = add_line_numbers(
        context.target
    )

    callee_context = (
        format_related_methods(
            title="DIRECT CALLEES",
            methods=context.callees,
        )
    )

    caller_context = (
        format_related_methods(
            title="DIRECT CALLERS",
            methods=context.callers,
        )
    )

    return f"""
TARGET METHOD
-------------
Name: {context.target.name}
Lines: {context.target.start_line}-{context.target.end_line}

{target_code}

{callee_context}

{caller_context}
""".strip()
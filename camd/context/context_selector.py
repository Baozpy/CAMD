from dataclasses import dataclass

from camd.context.method_extractor import JavaMethod


@dataclass
class SelectedContextMethod:
    method: JavaMethod
    relevance_score: float
    relation: str
    reason: str


class ContextSelector:

    def __init__(
        self,
        top_k_callees: int = 3,
        top_k_callers: int = 2,
    ):
        self.top_k_callees = top_k_callees
        self.top_k_callers = top_k_callers

    def select(
        self,
        target: JavaMethod,
        callees: list[JavaMethod],
        callers: list[JavaMethod],
    ) -> list[SelectedContextMethod]:

        candidates: list[SelectedContextMethod] = []

        for method in callees:
            score, reason = self._score_callee(
                target=target,
                method=method,
            )

            candidates.append(
                SelectedContextMethod(
                    method=method,
                    relevance_score=score,
                    relation="callee",
                    reason=reason,
                )
            )

        for method in callers:
            score, reason = self._score_caller(
                target=target,
                method=method,
            )

            candidates.append(
                SelectedContextMethod(
                    method=method,
                    relevance_score=score,
                    relation="caller",
                    reason=reason,
                )
            )

        selected_callees = sorted(
            [
                item
                for item in candidates
                if item.relation == "callee"
            ],
            key=lambda item: item.relevance_score,
            reverse=True,
        )[: self.top_k_callees]

        selected_callers = sorted(
            [
                item
                for item in candidates
                if item.relation == "caller"
            ],
            key=lambda item: item.relevance_score,
            reverse=True,
        )[: self.top_k_callers]

        return (
            selected_callees
            + selected_callers
        )

    def _score_callee(
        self,
        target: JavaMethod,
        method: JavaMethod,
    ) -> tuple[float, str]:

        score = 0.50
        reasons = [
            "Direct callee of the target method."
        ]

        target_name = target.name.lower()
        method_name = method.name.lower()

        shared_tokens = self._shared_name_tokens(
            target_name,
            method_name,
        )

        if shared_tokens:
            score += 0.15
            reasons.append(
                "Method name is semantically related "
                "to the target method."
            )

        suspicious_keywords = {
            "create",
            "parse",
            "convert",
            "decode",
            "validate",
            "check",
            "number",
            "integer",
            "long",
            "float",
            "double",
            "big",
        }

        if any(
            keyword in method_name
            for keyword in suspicious_keywords
        ):
            score += 0.15
            reasons.append(
                "Method participates in parsing, "
                "conversion, validation, or numeric logic."
            )

        target_size = (
            target.end_line
            - target.start_line
            + 1
        )

        method_size = (
            method.end_line
            - method.start_line
            + 1
        )

        if method_size <= 30:
            score += 0.05
            reasons.append(
                "Method is compact enough to provide "
                "focused semantic context."
            )

        if target_size > 100:
            score += 0.05
            reasons.append(
                "Target method is large, so direct "
                "helper context may clarify its behavior."
            )

        score = min(
            score,
            1.0,
        )

        return (
            score,
            " ".join(reasons),
        )

    def _score_caller(
        self,
        target: JavaMethod,
        method: JavaMethod,
    ) -> tuple[float, str]:

        score = 0.55

        reasons = [
            "Direct caller of the target method."
        ]

        shared_tokens = self._shared_name_tokens(
            target.name.lower(),
            method.name.lower(),
        )

        if shared_tokens:
            score += 0.15
            reasons.append(
                "Caller name is semantically related "
                "to the target method."
            )

        method_size = (
            method.end_line
            - method.start_line
            + 1
        )

        if method_size <= 40:
            score += 0.05
            reasons.append(
                "Caller provides compact usage context."
            )

        score = min(
            score,
            1.0,
        )

        return (
            score,
            " ".join(reasons),
        )

    @staticmethod
    def _shared_name_tokens(
        first: str,
        second: str,
    ) -> set[str]:

        keywords = {
            "number",
            "integer",
            "long",
            "float",
            "double",
            "big",
            "decimal",
            "parse",
            "create",
            "validate",
            "digit",
            "min",
            "max",
        }

        first_tokens = {
            token
            for token in keywords
            if token in first
        }

        second_tokens = {
            token
            for token in keywords
            if token in second
        }

        return (
            first_tokens
            & second_tokens
        )
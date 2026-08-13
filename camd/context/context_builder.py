from dataclasses import dataclass

from camd.context.call_extractor import (
    MethodCall,
    extract_method_calls,
)
from camd.context.method_extractor import (
    JavaMethod,
)


@dataclass
class MethodContext:
    target: JavaMethod
    callees: list[JavaMethod]
    callers: list[JavaMethod]


class ContextBuilder:

    def __init__(
        self,
        methods: list[JavaMethod],
    ):

        self.methods = methods

        self.methods_by_name: dict[
            str,
            list[JavaMethod],
        ] = {}

        for method in methods:

            self.methods_by_name.setdefault(
                method.name,
                [],
            ).append(
                method
            )

    def build(
        self,
        target: JavaMethod,
    ) -> MethodContext:

        callees = (
            self._find_callees(
                target
            )
        )

        callers = (
            self._find_callers(
                target
            )
        )

        return MethodContext(
            target=target,
            callees=callees,
            callers=callers,
        )

    @staticmethod
    def _is_local_call(
        call: MethodCall,
    ) -> bool:

        if call.qualifier is None:
            return True

        if call.qualifier == "this":
            return True

        return False

    def _resolve_call(
        self,
        call: MethodCall,
    ) -> JavaMethod | None:

        if not self._is_local_call(
            call
        ):
            return None

        candidates = (
            self.methods_by_name.get(
                call.callee,
                [],
            )
        )

        if not candidates:
            return None

        arity_matches = [
            method
            for method in candidates
            if (
                method.parameter_count
                == call.argument_count
            )
        ]

        if len(arity_matches) == 1:

            return arity_matches[0]

        # Ambiguous overload:
        # do not inject potentially incorrect context.
        if len(arity_matches) > 1:

            return None

        return None

    def _find_callees(
        self,
        target: JavaMethod,
    ) -> list[JavaMethod]:

        calls = extract_method_calls(
            target
        )

        callees: list[JavaMethod] = []

        seen = set()

        for call in calls:

            resolved = (
                self._resolve_call(
                    call
                )
            )

            if resolved is None:
                continue

            key = (
                resolved.name,
                resolved.start_line,
                resolved.end_line,
            )

            if key in seen:
                continue

            seen.add(
                key
            )

            callees.append(
                resolved
            )

        return callees

    def _find_callers(
        self,
        target: JavaMethod,
    ) -> list[JavaMethod]:

        callers: list[JavaMethod] = []

        seen = set()

        for method in self.methods:

            if method is target:
                continue

            calls = extract_method_calls(
                method
            )

            for call in calls:

                resolved = (
                    self._resolve_call(
                        call
                    )
                )

                if resolved is None:
                    continue

                if (
                    resolved.start_line
                    != target.start_line
                ):
                    continue

                if (
                    resolved.end_line
                    != target.end_line
                ):
                    continue

                key = (
                    method.name,
                    method.start_line,
                    method.end_line,
                )

                if key in seen:
                    break

                seen.add(
                    key
                )

                callers.append(
                    method
                )

                break

        return callers
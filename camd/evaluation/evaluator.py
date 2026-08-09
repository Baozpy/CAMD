from dataclasses import dataclass


@dataclass
class EvaluationMetrics:
    total: int

    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int

    accuracy: float
    precision: float
    recall: float
    f1: float

    localization_accuracy: float


def safe_divide(
    numerator: float,
    denominator: float,
) -> float:

    if denominator == 0:
        return 0.0

    return numerator / denominator


def evaluate_results(
    results: list[dict],
) -> EvaluationMetrics:

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    localization_correct = 0
    localization_total = 0

    for result in results:

        ground_truth = result["ground_truth"]
        prediction = result["prediction"]

        actual = ground_truth["is_defective"]
        predicted = prediction["is_defective"]

        if actual and predicted:
            true_positive += 1

        elif not actual and not predicted:
            true_negative += 1

        elif not actual and predicted:
            false_positive += 1

        elif actual and not predicted:
            false_negative += 1

        if actual:
            localization_total += 1

            predicted_line = (
                prediction["location"]["line"]
            )

            actual_line = ground_truth["buggy_line"]

            if (
                predicted
                and predicted_line == actual_line
            ):
                localization_correct += 1

    total = len(results)

    accuracy = safe_divide(
        true_positive + true_negative,
        total,
    )

    precision = safe_divide(
        true_positive,
        true_positive + false_positive,
    )

    recall = safe_divide(
        true_positive,
        true_positive + false_negative,
    )

    f1 = safe_divide(
        2 * precision * recall,
        precision + recall,
    )

    localization_accuracy = safe_divide(
        localization_correct,
        localization_total,
    )

    return EvaluationMetrics(
        total=total,
        true_positive=true_positive,
        true_negative=true_negative,
        false_positive=false_positive,
        false_negative=false_negative,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        localization_accuracy=localization_accuracy,
    )
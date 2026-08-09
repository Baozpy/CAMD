import json
from pathlib import Path

from camd.context.models import DefectSample


def load_dataset(
    metadata_path: Path,
) -> list[DefectSample]:

    samples = []

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)

            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line "
                    f"{line_number}: {line}"
                ) from exc

            sample = DefectSample(
                sample_id=item["sample_id"],
                file=item["file"],
                label=bool(item["label"]),
                defect_type=item.get(
                    "defect_type",
                    "none",
                ),
                buggy_line=int(
                    item.get("buggy_line", 0)
                ),
            )

            samples.append(sample)

    return samples
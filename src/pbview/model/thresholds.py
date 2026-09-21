"""Threshold profile serialization and parsing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from ..config import SCHEMA_VERSION


@dataclass(frozen=True)
class SampleSetThresholds:
    missing_cutoff: int
    lower_coverage: int
    upper_coverage: int
    max_missing_samples: int


@dataclass(frozen=True)
class ThresholdProfile:
    schema_version: int
    dataset_id: str
    sample_sets: dict[str, SampleSetThresholds]

    def to_yaml(self) -> str:
        return yaml.safe_dump(
            {
                "schema_version": self.schema_version,
                "dataset_id": self.dataset_id,
                "sample_sets": {s: asdict(t) for s, t in self.sample_sets.items()},
            },
            sort_keys=False,
        )

    def write(self, path: str | Path) -> None:
        Path(path).write_text(self.to_yaml())

    @classmethod
    def from_yaml(cls, text: str) -> "ThresholdProfile":
        data = yaml.safe_load(text)
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema version: {data.get('schema_version')}"
            )
        sets = {s: SampleSetThresholds(**t) for s, t in data["sample_sets"].items()}
        return cls(
            schema_version=data["schema_version"],
            dataset_id=data["dataset_id"],
            sample_sets=sets,
        )

    @classmethod
    def read(cls, path: str | Path) -> "ThresholdProfile":
        return cls.from_yaml(Path(path).read_text())


def profile_from_state(state, datastore_id: str) -> ThresholdProfile:
    """Build a ThresholdProfile from a SelectionState instance."""
    sample_sets = {}
    for s in state.coord.sample_set_names_all:
        sample_sets[s] = SampleSetThresholds(
            missing_cutoff=int(getattr(state, "missing_cutoff")),
            lower_coverage=int(getattr(state, f"lower_coverage_{s}")),
            upper_coverage=int(getattr(state, f"upper_coverage_{s}")),
            max_missing_samples=int(getattr(state, f"max_missing_samples_{s}")),
        )
    return ThresholdProfile(
        schema_version=SCHEMA_VERSION,
        dataset_id=datastore_id,
        sample_sets=sample_sets,
    )

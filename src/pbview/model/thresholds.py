"""Threshold profile serialization and parsing."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from ..config import SCHEMA_VERSION
from .coordinates import Coordinates

if TYPE_CHECKING:
    from .track import Track


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

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "sample_sets": {s: asdict(t) for s, t in self.sample_sets.items()},
        }

    @classmethod
    def read(cls, path: str | Path) -> "ThresholdProfile":
        return cls.from_yaml(Path(path).read_text())


def profile_from_state(state, datastore_id: str) -> ThresholdProfile:
    """Build a ThresholdProfile from a SelectionState instance."""
    sample_sets = {}
    for s in state.track().groups("sample_set"):
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


def default_max_missing_samples(n_samples: int) -> int:
    """Default number of samples with missing data that a site may have.

    Half of the samples in the sample set; used by the app and by
    `profile_from_defaults`.
    """
    return int(n_samples / 2)


def profile_from_defaults(
    track: Track, defaults: Mapping[str, int], datastore_id: str, missing_cutoff: int
) -> ThresholdProfile:
    """Build a ThresholdProfile from the default thresholds of each sample set.

    `defaults` is the output of `compute_threshold_defaults`. The profile
    holds the thresholds that the app starts with, so exporting from the
    command line gives the same values as exporting from the app before
    any change is made.
    """
    sample_sets = {}
    for s in track.groups("sample_set"):
        n_samples = track.n_selected(Coordinates().with_sample_sets([s]).column)
        sample_sets[s] = SampleSetThresholds(
            missing_cutoff=int(missing_cutoff),
            lower_coverage=int(defaults[f"lower_coverage_{s}"]),
            upper_coverage=int(defaults[f"upper_coverage_{s}"]),
            max_missing_samples=default_max_missing_samples(n_samples),
        )
    return ThresholdProfile(
        schema_version=SCHEMA_VERSION,
        dataset_id=datastore_id,
        sample_sets=sample_sets,
    )

from __future__ import annotations

import pytest
import yaml

from pbview.model.thresholds import (
    SCHEMA_VERSION,
    SampleSetThresholds,
    ThresholdProfile,
    profile_from_state,
)


@pytest.fixture
def sample_profile():
    return ThresholdProfile(
        schema_version=SCHEMA_VERSION,
        dataset_id="abc123",
        sample_sets={
            "ALL": SampleSetThresholds(3, 5, 500, 3),
            "set_A": SampleSetThresholds(3, 4, 200, 3),
            "set_B": SampleSetThresholds(3, 6, 300, 5),
        },
    )


def test_roundtrip_yaml(sample_profile):
    text = sample_profile.to_yaml()
    parsed = ThresholdProfile.from_yaml(text)
    assert parsed == sample_profile


def test_roundtrip_file(sample_profile, tmp_path):
    p = tmp_path / "thresholds.yml"
    sample_profile.write(p)
    parsed = ThresholdProfile.read(p)
    assert parsed == sample_profile


def test_yaml_is_stable_and_ordered(sample_profile):
    text = sample_profile.to_yaml()
    data = yaml.safe_load(text)
    assert list(data.keys()) == ["schema_version", "dataset_id", "sample_sets"]
    assert list(data["sample_sets"]) == ["ALL", "set_A", "set_B"]


def test_schema_version_mismatch_raises(sample_profile):
    text = sample_profile.to_yaml().replace(
        f"schema_version: {SCHEMA_VERSION}",
        f"schema_version: {SCHEMA_VERSION + 99}",
    )
    with pytest.raises(ValueError, match="Unsupported schema version"):
        ThresholdProfile.from_yaml(text)


def test_missing_field_raises(sample_profile):
    text = sample_profile.to_yaml()
    # remove upper_coverage from set_A
    data = yaml.safe_load(text)
    del data["sample_sets"]["set_A"]["upper_coverage"]
    with pytest.raises(TypeError):
        ThresholdProfile.from_yaml(yaml.safe_dump(data))


class _Coordinates:
    def __init__(self):
        self.sample_set_names_all = ["ALL", "set_A"]


class _FakeState:
    """Minimal stand-in for SelectionState in tests."""

    def __init__(self):
        self.missing_cutoff = 3
        self.max_missing_samples_ALL = 3
        self.max_missing_samples_set_A = 3
        self.lower_coverage_ALL = 5
        self.upper_coverage_ALL = 500
        self.lower_coverage_set_A = 4
        self.upper_coverage_set_A = 200
        self.coord = _Coordinates()


def test_profile_from_state_builds_correct_profile():
    state = _FakeState()
    profile = profile_from_state(state, datastore_id="xyz")
    assert profile.dataset_id == "xyz"
    assert profile.schema_version == SCHEMA_VERSION
    assert profile.sample_sets["ALL"] == SampleSetThresholds(3, 5, 500, 3)
    assert profile.sample_sets["set_A"] == SampleSetThresholds(3, 4, 200, 3)

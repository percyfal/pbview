import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
SCRIPT = ROOT / ".github" / "scripts" / "check_release_version.py"


@pytest.fixture(scope="module")
def check():
    spec = importlib.util.spec_from_file_location("check_release_version", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def wheel(dist, version):
    (dist / f"pbview-{version}-py3-none-any.whl").touch()


@pytest.mark.parametrize(
    ("tag", "version"),
    [("0.1.0-alpha.3", "0.1.0a3"), ("v0.1.0a3", "0.1.0a3"), ("1.2.0", "1.2.0")],
)
def test_the_tag_and_the_wheel_agree(
    check, tmp_path, monkeypatch, capsys, tag, version
):
    wheel(tmp_path, version)
    monkeypatch.setenv("TAG", tag)
    check.main(str(tmp_path))
    assert f"is version {version}" in capsys.readouterr().out


@pytest.mark.parametrize("version", ["0.1.0a2", "0.1.0a4.dev1+gd6a79a4e7"])
def test_another_version_is_refused(check, tmp_path, monkeypatch, version):
    wheel(tmp_path, version)
    monkeypatch.setenv("TAG", "0.1.0-alpha.3")
    with pytest.raises(SystemExit, match="the tag is 0.1.0a3"):
        check.main(str(tmp_path))


def test_there_must_be_exactly_one_wheel(check, tmp_path, monkeypatch):
    monkeypatch.setenv("TAG", "0.1.0-alpha.3")
    with pytest.raises(SystemExit, match="expected one wheel"):
        check.main(str(tmp_path))
    wheel(tmp_path, "0.1.0a3")
    wheel(tmp_path, "0.1.0a2")
    with pytest.raises(SystemExit, match="expected one wheel"):
        check.main(str(tmp_path))

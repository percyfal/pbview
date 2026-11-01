"""Check that the built wheel has the version of the release tag.

The version comes from the git tag (hatch-vcs). Building anything but the
tagged commit gives a dev version, which must not be attached to a release.

Usage: TAG=MAJOR.MINOR.PATCH python check_release_version.py [dist-directory]
"""

import os
import sys
from pathlib import Path

from packaging.version import Version


def main(dist: str = "dist") -> None:
    tag = Version(os.environ["TAG"])
    wheels = sorted(Path(dist).glob("pbview-*.whl"))
    if len(wheels) != 1:
        sys.exit(f"expected one wheel in {dist}, found {[w.name for w in wheels]}")
    built = Version(wheels[0].name.split("-")[1])
    if built != tag:
        sys.exit(f"{wheels[0].name} is version {built}, the tag is {tag}")
    print(f"{wheels[0].name} is version {built}, as the tag")


if __name__ == "__main__":
    main(*sys.argv[1:])

# Contributing

WIP.

## Releases

The version comes from the git tag (`hatch-vcs`). The package is pure
Python, so one wheel, `pbview-<version>-py3-none-any.whl`, serves
every platform; its dependencies, including `pbzarr`, are installed
from PyPI.

To attach a wheel and an sdist to a GitHub release, publish the
release: the `Release` workflow builds them from the tag and uploads
them. To add them to a release that already exists, run the workflow
by hand (Actions, Release, Run workflow) and give the tag. The
workflow refuses to upload a build whose version is not that of the
tag.

To build locally, check out the tag and run `pixi r build`; the files
are written to `dist/`. Anything but a tagged commit gives a dev
version.

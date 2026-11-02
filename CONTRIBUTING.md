# Contributing

WIP.

## Releases

The version comes from the git tag (`hatch-vcs`): tag `0.1.0-alpha.3`
gives version `0.1.0a3`. The package is pure Python, so one wheel,
`pbview-<version>-py3-none-any.whl`, serves every platform; its
dependencies, including `pbzarr`, are installed from PyPI.

Publishing a GitHub release runs the `Release` workflow, which builds
the wheel and an sdist from the tag, refuses a build whose version is
not that of the tag and attaches both files to the release. It also
uploads them to PyPI once the repository variable `PUBLISH_TO_PYPI` is
`true`. To build the files locally, check out the tag and run
`pixi r build`; they are written to `dist/`. Anything but a tagged
commit gives a dev version.

Run the workflow by hand (Actions, Release, Run workflow) to add files
to a release that already exists. Give its tag and choose where to
publish: `none` (only attach to the release), `testpypi` or `pypi`.

PyPI is told to trust the workflow (trusted publishing), so no token is
stored. Set this up once:

1. On [pypi.org](https://pypi.org/manage/account/publishing/), add a
   pending publisher: project `pbview`, owner `percyfal`, repository
   `pbview`, workflow `release.yml`, environment `pypi`. Do the same on
   [test.pypi.org](https://test.pypi.org/manage/account/publishing/)
   with the environment `testpypi`. Both need an account first.
2. On GitHub (Settings, Environments) create the environments `pypi`
   and `testpypi`. Requiring a reviewer for `pypi` makes every release
   wait for an approval before it is uploaded.
3. Try the first upload on TestPyPI: run the workflow by hand with
   `publish` set to `testpypi`, then install from there with
   `pip install --index-url https://test.pypi.org/simple/
   --extra-index-url https://pypi.org/simple/ pbview==<version>`.
    A version can only be uploaded once and the files of a version
    cannot be replaced, so check the version before publishing; a
    release can be withdrawn (yanked) on PyPI but not reused. Alpha
    versions are not installed unless the version is given
    (`pbview==0.1.0a4`) or `--pre` is used.
4. When the first upload has worked, set the repository variable
   `PUBLISH_TO_PYPI` to `true` (Settings, Secrets and variables,
   Actions, Variables). From then on every published release is uploaded
   to PyPI. Until it is set, releases only get the wheel and the sdist
   attached, and PyPI is published to by hand.

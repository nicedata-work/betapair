# Releasing

Releases are built and uploaded by [`.github/workflows/release.yml`](.github/workflows/release.yml)
on a `v*` tag push, using [PyPI trusted publishing] (OIDC) — no API token is
stored in the repository.

## One-time setup

1. **PyPI** — because `betapair` does not exist yet, register a *pending*
   trusted publisher at <https://pypi.org/manage/account/publishing/> with:

   | field | value |
   | :--- | :--- |
   | PyPI project name | `betapair` |
   | Owner | `nicedata-work` |
   | Repository name | `betapair` |
   | Workflow name | `release.yml` |
   | Environment name | `pypi` |

   After the first upload the publisher becomes an ordinary one and can be
   edited under the project's *Publishing* settings.

2. **GitHub** — create an environment named `pypi` in *Settings →
   Environments*. Nothing needs to be stored in it; the name is what PyPI
   checks against the OIDC token.

## Cutting a release

1. Move the `[Unreleased]` entries in `CHANGELOG.md` under a new
   `## [x.y.z] - YYYY-MM-DD` heading and update the two link references at the
   bottom of the file.
2. Set the same version in `pyproject.toml` (`[project] version`). The
   workflow fails if the tag and this value disagree, so this step cannot be
   skipped silently.
3. Confirm `main` is green — the `calibration` job is what keeps
   `docs/validation.md` in step with the code.
4. Commit, then tag and push:

   ```bash
   git commit -am "release x.y.z"
   git tag -a vx.y.z -m "vx.y.z"
   git push origin main vx.y.z
   ```

The tag runs the workflow: `check` (lint, format, types, tests) → `build`
(`uv build`, `twine check`) → `publish` (trusted publishing, with PEP 740
attestations) → `github-release` (release notes taken from the `CHANGELOG.md`
section for that version).

`workflow_dispatch` runs only `check` and `build` — it never publishes, since
there is no tag to publish and re-uploading a version is rejected by PyPI.

## Notes

- **PyPI is append-only per version.** A file can be deleted within a few
  hours of upload, but the version number itself cannot be reused, so a broken
  release is fixed by publishing the next patch version, not by re-tagging.
  To rehearse the whole flow first, point `pypa/gh-action-pypi-publish` at
  `repository-url: https://test.pypi.org/legacy/` and register the same
  trusted publisher on TestPyPI.
- The build job uploads the exact artifacts the publish job downloads, so what
  is checked is what is uploaded.

[PyPI trusted publishing]: https://docs.pypi.org/trusted-publishers/

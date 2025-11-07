# Contribution Guide

The main branch is the trunk and features branches are squashed merge after a successful Pull request.

## Contributing a feature

- [X] make sure the tests pass locally
  - `pytest ./src/tests/unit`  (approx 1 min.)
  - `pytest ./src/tests/integration` (approx 3 mins if data is available)
- [X] use `ruff format` and `ruff check` to make sure the formatting is correct
- [X] [CHANGELOG.md](CHANGELOG.md) documents the changes, references the PR, the date and the new version number in `setup.py`
- [X] create a PR from your feature branch to main

### Reviewer steps for a feature PR
- [X] the CI passes
- [X] squash-merge upon a successful review

## Release

- [X] create tag corresponding to the version number `X.Y.Z` on the `main` branch

```shell
tag=X.Y.Z
git tag -a $tag 
git push origin $tag
```

- [X] Create new release with tag `X.Y.Z` (will automatically publish to PyPI).

```shell 
gh release create $tag
```

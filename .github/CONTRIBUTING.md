See the [Scientific Python Developer Guide][spc-dev-intro] for a detailed
description of best practices for developing scientific packages.

[spc-dev-intro]: https://learn.scientific-python.org/development/

# Quick development

The fastest way to start with development is to use nox. If you don't have nox,
you can use `uvx nox` to run it without installing, or `uv tool install nox`. If
you don't have uv, you can
[install it a variety of ways](https://docs.astral.sh/uv/getting-started/installation/),
including with pip, pipx, brew, and just downloading the binary (single file).

To use, run `nox`. This will lint and test using every installed version of
Python on your system, skipping ones that are not installed. You can also run
specific jobs:

```console
$ nox -s lint  # Lint only
$ nox -s tests  # Python tests
$ nox -s docs  # Build and serve the docs
$ nox -s build  # Make an SDist and wheel
```

Nox handles everything for you, including setting up an temporary virtual
environment for each run.

# Setting up a development environment manually

You can set up a development environment by running:

```bash
uv sync
```

# Pre-commit

You should prepare pre-commit or prek, which will help you by checking that
commits pass required checks:

```bash
uv tool install pre-commit # or brew install pre-commit on macOS
pre-commit install # Will install a pre-commit hook into the git repo
```

You can also/alternatively run `pre-commit run` (changes only) or
`pre-commit run --all-files` to check even without installing the hook.

# Testing

Use pytest to run the unit checks:

```bash
uv run pytest                 # every test
uv run pytest -m "not slow"   # without the slow ones, as most CI jobs of a pull request
```

A test may take 10 s on the CI job that runs every test; a slower one fails
there unless it is marked `@pytest.mark.slow`. Prefer proving the behaviour on a
smaller model. Slow tests run on that one job of a pull request, and on every
job of `main`.

# Coverage

Use pytest-cov to generate coverage reports:

```bash
uv run pytest --cov=magpylib-material-response
```

# Building docs

You can build and serve the docs using:

```bash
nox -s docs
```

You can build the docs only with:

```bash
nox -s docs --non-interactive
```

The CI docs job builds the docs as Read the Docs does and checks the time of
each page against `docs/check_budget.py`: 45 s a page on its runner, 480 s in
all. A page that needs longer stores its slow solves from a study script, as the
examples do.

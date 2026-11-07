---
html_theme.sidebar_secondary.remove: true
---

# magpylib-material-response

<!-- the scrolling story, data from landing_data.py; not in the Markdown twin -->

````{only} not sphinx_llm_markdown
```{raw} html
:file: _static/landing/landing.html
```
````

## In short

```{include} ../README.md
:start-after: <!-- SPHINX-START -->
:end-before: <!-- SPHINX-END -->
```

## Installation

```{include} _install.md

```

## Where to start

- **New here:** [Getting started](getting_started.md) solves one model with each
  of the three solvers, in a few minutes.
- **Which solver for my model:** [Choosing a solver](guide/choosing.md).
- **How it works, in detail:** [Inside the solvers](inside/index.md), with an
  interactive map of each solver and the papers behind every step.
- **Worked models against finite elements:** the [examples](examples/index.md).
- **With a coding agent:** [Getting started](getting_started.md) has a prompt to
  paste; every page also exists as Markdown, indexed in
  [llms.txt](https://magpylib-material-response.readthedocs.io/en/latest/llms.txt).

```{toctree}
:maxdepth: 1
:hidden:

getting_started
guide/index
inside/index
examples/index
reference
```

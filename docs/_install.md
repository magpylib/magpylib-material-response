`````{tab-set}
:sync-group: installer

````{tab-item} uv
:sync: uv

```bash
uv add magpylib-material-response
uv add "magpylib-material-response[fem]"     # finite elements, to check results
uv add "magpylib-material-response[tetgen]"  # tetrahedral cells for any shape
```

````

````{tab-item} pip
:sync: pip

```bash
pip install magpylib-material-response
pip install "magpylib-material-response[fem]"     # finite elements, to check results
pip install "magpylib-material-response[tetgen]"  # tetrahedral cells for any shape
```

````

````{tab-item} Latest from GitHub
:sync: git

```bash
uv add "magpylib-material-response @ git+https://github.com/magpylib/magpylib-material-response"
pip install "git+https://github.com/magpylib/magpylib-material-response"
```

````

`````

The `fem` extra installs NGSolve (LGPL-2.1), which has no wheels for Linux on
ARM or for musl.

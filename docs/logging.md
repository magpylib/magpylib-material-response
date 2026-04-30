# Logging Configuration

The magpylib-material-response package uses
[Loguru](https://loguru.readthedocs.io/) to emit informative messages about
computation progress and debugging information.

## Default Behavior

Following the
[recommended pattern for libraries](https://loguru.readthedocs.io/en/stable/resources/recipes.html#configuring-loguru-to-be-used-by-a-library-or-an-application),
the package is **disabled at import time** and produces no output until the
user opts in. Importing the package never adds, removes or replaces any
loguru sinks already configured by the host application.

## Enabling Logging

To see log messages from the library, call `configure_logging()`:

```python
from magpylib_material_response import configure_logging
from magpylib_material_response.demag import apply_demag

# Enable logging with default settings (INFO level, colored output to stderr)
configure_logging()
```

`configure_logging()` adds a sink that only receives records emitted from
within the `magpylib_material_response` package. It is safe to call multiple
times; each call replaces only the sink it added previously, leaving any
other loguru sinks untouched.

```{note}
By default, loguru ships with a pre-installed `sys.stderr` sink (handler id
`0`). Because `configure_logging()` does not touch sinks it did not add, you
may see each package record appear **twice**: once via loguru's default
handler and once via the sink added by `configure_logging`. To avoid this,
remove the default handler before configuring:

    from loguru import logger
    logger.remove(0)
    configure_logging()
```

## Configuration Options

### Log Level

```python
configure_logging(level="DEBUG")    # detailed internal operations
configure_logging(level="WARNING")  # only warnings and errors
# Available: DEBUG, INFO, WARNING, ERROR, CRITICAL
```

### Output Destination

```python
import sys

configure_logging(sink=sys.stdout)        # stdout instead of stderr
configure_logging(sink="/path/to/file.log")  # log file
```

### Disable Colors and Time

```python
configure_logging(enable_colors=False)  # useful for log files
configure_logging(show_time=False)
```

### Step Timing Threshold

The library wraps long-running steps in a `timelog` context that emits a
"Completed: ..." record when the step exceeds a duration threshold. The
default is `1.0` second.

```python
# Log every step, regardless of duration:
configure_logging(min_log_time=0)

# Only log very long steps:
configure_logging(min_log_time=10)
```

This setting is also picked up by `apply_demag`, `demag_tensor`,
`filter_distance`, and `match_pairs` whenever their own `min_log_time`
argument is left at its default (`None`). Passing an explicit value to those
functions still overrides the global setting for that call.

## Environment Variables

```bash
export MAGPYLIB_LOG_LEVEL=DEBUG
export MAGPYLIB_LOG_COLORS=false
export MAGPYLIB_LOG_TIME=false
export MAGPYLIB_LOG_MIN_TIME=0
```

## Disabling Logging

```python
from magpylib_material_response import disable_logging

disable_logging()
```

This re-disables the package and removes only the sink previously added by
`configure_logging`.

## Custom Application-Side Configuration

If your application already configures loguru, you do not need to call
`configure_logging()` at all. Instead, simply enable the package and let your
own sinks handle the records:

```python
from loguru import logger

logger.enable("magpylib_material_response")
# your existing logger.add(...) sinks now also receive package records
```

## Example Usage

```python
import magpylib as magpy
from magpylib_material_response import configure_logging
from magpylib_material_response.demag import apply_demag
from magpylib_material_response.meshing import mesh_Cuboid

configure_logging(level="INFO")

magnet = magpy.magnet.Cuboid(dimension=(0.01, 0.01, 0.02), polarization=(0, 0, 1))
magnet.susceptibility = 0.1

meshed = mesh_Cuboid(magnet, target_elems=1000, verbose=True)
apply_demag(meshed, inplace=True)
```

## See Also

- {doc}`examples/index` - Working examples that demonstrate logging output
- [Loguru Documentation](https://loguru.readthedocs.io/) - Complete reference
  for the underlying logging library

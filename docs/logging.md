# Logging Configuration

The magpylib-material-response package uses structured logging with [Loguru](https://loguru.readthedocs.io/) to provide informative messages about computation progress and debugging information.

## Default Behavior

By default, the library **does not output any log messages**. This follows best practices for Python libraries to avoid cluttering user output unless explicitly requested.

## Enabling Logging

To see log messages from the library, you need to configure logging:

```python
from magpylib_material_response import configure_logging
from magpylib_material_response.demag import apply_demag

# Enable logging with default settings (INFO level, colored output to stderr)
configure_logging()

# Now use the library - you'll see progress messages
# ... your code here
```

## Configuration Options

### Log Level
```python
from magpylib_material_response import configure_logging

# Set to DEBUG for detailed internal operations
configure_logging(level="DEBUG")

# Set to WARNING to only see important warnings and errors
configure_logging(level="WARNING")

# Available levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
```

### Output Destination
```python
import sys
from magpylib_material_response import configure_logging

# Output to stdout instead of stderr
configure_logging(sink=sys.stdout)

# Output to a file
configure_logging(sink="/path/to/logfile.log")
```

### Disable Colors and Time
```python
from magpylib_material_response import configure_logging

# Disable colored output (useful for log files)
configure_logging(enable_colors=False)

# Disable timestamps
configure_logging(show_time=False)
```

## Environment Variables

You can also configure logging using environment variables:

```bash
# Set log level
export MAGPYLIB_LOG_LEVEL=DEBUG

# Disable colors
export MAGPYLIB_LOG_COLORS=false

# Disable timestamps
export MAGPYLIB_LOG_TIME=false
```

## Disabling Logging

To completely disable logging output:

```python
from magpylib_material_response import disable_logging

disable_logging()
```

## Example Usage

```python
import magpylib as magpy
from magpylib_material_response import configure_logging
from magpylib_material_response.demag import apply_demag
from magpylib_material_response.meshing import mesh_Cuboid

# Enable logging to see progress
configure_logging(level="INFO")

# Create a magnet
magnet = magpy.magnet.Cuboid(
    dimension=(0.01, 0.01, 0.02),
    polarization=(0, 0, 1)
)
magnet.susceptibility = 0.1

# Mesh the magnet - you'll see meshing progress
meshed = mesh_Cuboid(magnet, target_elems=1000, verbose=True)

# Apply demagnetization - you'll see computation progress
apply_demag(meshed, inplace=True)
```

This will output structured log messages showing the progress of operations, timing information, and any warnings or errors.

## See Also

- {doc}`examples/index` - Working examples that demonstrate logging output
- [Loguru Documentation](https://loguru.readthedocs.io/) - Complete reference for the underlying logging library
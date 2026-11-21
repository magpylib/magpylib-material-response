"""What the reference files share (#208): their JSON layout, the encoding of
floats and the tolerances' floor, margin and rounding. Imported by the
scripts next to it."""

import json
import platform

import numpy as np

RTOL = 1e-12  # the floor of every tolerance, relative
LOOSE = 1e-6  # a tolerance looser than this, relative, needs a reason
MARGIN = 16  # tolerance over the numpy implementation's own measured error
ISSUE = "https://github.com/magpylib/magpylib-material-response/issues/208"
ENCODING = (
    "floats as shortest round-trip decimals, to be parsed correctly rounded; "
    "'nan', 'inf', '-inf' as strings"
)


def platform_name():
    """The platform, whose libm decides numpy's last bits."""
    return f"{platform.system()} {platform.machine()}"


def decade(x):
    """``x`` rounded up to a power of ten: a tolerance never shrinks, and a
    last-bit difference between platforms' libm does not change the file."""
    if x == 0 or not np.isfinite(x):
        return float(x)
    return float(f"1e{int(np.ceil(np.log10(x)))}")


def encode(a):
    a = np.asarray(a, float)
    if np.all(np.isfinite(a)):
        return a.tolist()
    flat = [x if np.isfinite(x) else ("nan" if np.isnan(x) else ("inf" if x > 0 else "-inf"))
            for x in a.ravel().tolist()]  # fmt: skip
    return np.array(flat, object).reshape(a.shape).tolist()


def clean(x):
    """Values rounded to 15 significant digits: the decimals a person wrote,
    0.0045 rather than the product's 0.0045000000000000005."""
    return np.array([float(f"{v:.15g}") for v in np.ravel(x)]).reshape(np.shape(x))


def dump(obj, depth=0):
    """JSON with every list of numbers on one line: one row per line, so that a
    diff shows which value moved. Floats are written as ``repr``, exact."""
    pad, inner = " " * depth, " " * (depth + 1)
    if isinstance(obj, dict):
        items = [
            f"{inner}{json.dumps(k)}: {dump(v, depth + 1)}" for k, v in obj.items()
        ]
        return "{\n" + ",\n".join(items) + f"\n{pad}}}" if items else "{}"
    if isinstance(obj, list):
        # numbers on one line (a vector, a row); texts and nested lists one per line
        if all(isinstance(v, int | float) or v in ("nan", "inf", "-inf") for v in obj):
            return json.dumps(obj, separators=(", ", ": "))
        return "[\n" + ",\n".join(inner + dump(v, depth + 1) for v in obj) + f"\n{pad}]"
    return json.dumps(obj)

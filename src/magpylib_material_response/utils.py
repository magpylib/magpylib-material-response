from __future__ import annotations

import json
import threading
import time
import warnings
from contextlib import contextmanager

import magpylib as magpy
from loguru import logger
from magpylib._src.obj_classes.class_BaseExcitations import BaseCurrent, BaseMagnet
from scipy.spatial.transform import Rotation

from magpylib_material_response import logging_config


class ElapsedTimeThread(threading.Thread):
    """ "Stoppable thread that logs the time elapsed"""

    def __init__(self, msg=None, min_log_time=None):
        super().__init__()
        self._stop_event = threading.Event()
        self.thread_start = time.time()
        self.msg = msg
        self.min_log_time = (
            logging_config.DEFAULT_MIN_LOG_TIME
            if min_log_time is None
            else min_log_time
        )
        self._msg_displayed = False

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

    def getStart(self):
        return self.thread_start

    def run(self):
        self.thread_start = time.time()
        while not self.stopped():
            if (
                self.msg is not None
                and time.time() - self.thread_start > self.min_log_time
                and not self._msg_displayed
            ):
                logger.info("Starting: {operation}", operation=self.msg)
                self._msg_displayed = True
            # include a delay here so the thread doesn't uselessly thrash the CPU
            time.sleep(max(0.01, self.min_log_time / 5))


def format_duration(seconds):
    """Format a duration in seconds using an appropriate unit.

    Picks ns / µs / ms / s / m s / h m s based on magnitude so values
    spanning many orders of magnitude stay readable.
    """
    if seconds < 0:
        return f"-{format_duration(-seconds)}"
    if seconds < 1e-6:
        return f"{seconds * 1e9:.0f} ns"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.3g} µs"
    if seconds < 1.0:
        return f"{seconds * 1e3:.3g} ms"
    if seconds < 60.0:
        return f"{seconds:.3g} s"
    if seconds < 3600.0:
        m, s = divmod(seconds, 60)
        return f"{int(m)} m {s:04.1f} s"
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)} h {int(m):02d} m {int(s):02d} s"


@contextmanager
def timelog(msg, min_log_time=None):
    """Measure and log time with loguru as context manager.

    If ``min_log_time`` is None, the value set by
    :func:`magpylib_material_response.configure_logging` (default ``1.0`` s)
    is used.
    """
    if min_log_time is None:
        min_log_time = logging_config.DEFAULT_MIN_LOG_TIME
    start = time.perf_counter()
    thread_timer = ElapsedTimeThread(msg=msg, min_log_time=min_log_time)
    thread_timer.start()
    try:
        yield
    except Exception:
        logger.opt(exception=True).error("Failed: {operation}", operation=msg)
        raise
    else:
        end = time.perf_counter() - start
        if end > min_log_time:
            logger.info(
                "Completed: {operation} in {duration}",
                operation=msg,
                duration=format_duration(end),
            )
    finally:
        thread_timer.stop()
        thread_timer.join()


def _serialize_recursive(obj, parent="warn"):
    """Serialize a magpylib object to a JSON-compatible dict.

    Supported classes:
        - ``magpylib.magnet.Cuboid``, ``Cylinder``, ``CylinderSegment``
        - ``magpylib.current.Polyline``, ``Circle``
        - ``magpylib.Sensor``
        - ``magpylib.Collection`` (recursively)

    The output uses namespaced ``type`` discriminators (e.g. ``"magnet.Cuboid"``,
    ``"current.Polyline"``) and is composed entirely of JSON primitives.
    """
    typ = _TYPE_BY_CLASS.get(type(obj))
    if typ is None:
        msg = (
            f"Unsupported object type: {obj.__class__.__name__!r}. "
            f"Supported types: {sorted(_CLASS_BY_TYPE)}"
        )
        raise TypeError(msg)

    dd = {
        "type": typ,
        "position": {"value": obj.position.tolist(), "unit": "m"},
        "orientation": {
            "value": obj.orientation.as_matrix().tolist(),
            "representation": "matrix",
        },
    }
    if getattr(obj, "_style", None) is not None or obj._style_kwargs:
        dd["style"] = obj.style.as_dict()
    if parent == "warn" and obj.parent is not None:
        warnings.warn(
            f"object parent ({obj.parent}) not included in serialization", stacklevel=2
        )

    if isinstance(obj, BaseMagnet):
        dd["polarization"] = {"value": obj.polarization.tolist(), "unit": "T"}
        susceptibility = getattr(obj, "susceptibility", None)
        if susceptibility is not None:
            dd["susceptibility"] = {"value": susceptibility}

    if isinstance(obj, BaseCurrent):
        dd["current"] = {"value": float(obj.current), "unit": "A"}

    if isinstance(obj, magpy.magnet.Cuboid | magpy.magnet.Cylinder):
        dd["dimension"] = {"value": obj.dimension.tolist(), "unit": "m"}
    elif isinstance(obj, magpy.magnet.CylinderSegment):
        # (r1, r2, h, phi1, phi2) — first three in m, last two in deg.
        dd["dimension"] = {"value": obj.dimension.tolist(), "unit": "m,m,m,deg,deg"}
    elif isinstance(obj, magpy.current.Polyline):
        dd["vertices"] = {"value": obj.vertices.tolist(), "unit": "m"}
    elif isinstance(obj, magpy.current.Circle):
        dd["diameter"] = {"value": float(obj.diameter), "unit": "m"}
    elif isinstance(obj, magpy.Sensor):
        if obj.pixel is not None:
            dd["pixel"] = {"value": obj.pixel.tolist(), "unit": "m"}
    elif isinstance(obj, magpy.Collection):
        dd["children"] = [
            _serialize_recursive(child, parent="ignore") for child in obj.children
        ]
    return dd


def _check_unit(field_name, inp, expected):
    got = inp.get("unit")
    if got != expected:
        msg = f"{field_name} unit must be {expected!r}, got {got!r}"
        raise ValueError(msg)


def _deserialize_recursive(inp):
    """Deserialize a dict produced by :func:`_serialize_recursive`."""
    typ = inp.get("type")
    constr = _CLASS_BY_TYPE.get(typ)
    if constr is None:
        msg = f"Unknown type tag {typ!r}. Supported tags: {sorted(_CLASS_BY_TYPE)}"
        raise TypeError(msg)

    kw = {}

    # position
    _check_unit("Position", inp["position"], "m")
    kw["position"] = inp["position"]["value"]

    # orientation
    rep = inp["orientation"].get("representation")
    if rep != "matrix":
        msg = f"Orientation representation must be 'matrix', got {rep!r}"
        raise ValueError(msg)
    kw["orientation"] = Rotation.from_matrix(inp["orientation"]["value"])

    style = inp.get("style")
    if style is not None:
        kw["style"] = style

    if inp.get("parent") is not None:
        warnings.warn(f"object parent ({inp['parent']}) ignored", stacklevel=2)

    if issubclass(constr, BaseMagnet):
        _check_unit("Polarization", inp["polarization"], "T")
        kw["polarization"] = inp["polarization"]["value"]

    if issubclass(constr, BaseCurrent):
        _check_unit("Current", inp["current"], "A")
        kw["current"] = inp["current"]["value"]

    if constr in (magpy.magnet.Cuboid, magpy.magnet.Cylinder):
        _check_unit("Dimension", inp["dimension"], "m")
        kw["dimension"] = inp["dimension"]["value"]
    elif constr is magpy.magnet.CylinderSegment:
        _check_unit("Dimension", inp["dimension"], "m,m,m,deg,deg")
        kw["dimension"] = inp["dimension"]["value"]
    elif constr is magpy.current.Polyline:
        _check_unit("Vertices", inp["vertices"], "m")
        kw["vertices"] = inp["vertices"]["value"]
    elif constr is magpy.current.Circle:
        _check_unit("Diameter", inp["diameter"], "m")
        kw["diameter"] = inp["diameter"]["value"]
    elif constr is magpy.Sensor:
        if "pixel" in inp:
            _check_unit("Pixel", inp["pixel"], "m")
            kw["pixel"] = inp["pixel"]["value"]

    obj = constr(**kw)

    if inp.get("susceptibility") is not None:
        obj.susceptibility = inp["susceptibility"]["value"]

    if constr is magpy.Collection:
        obj.add(*[_deserialize_recursive(child) for child in inp["children"]])

    return obj


# Type-tag <-> class mapping. Namespaced tags avoid ambiguity if magpylib ever
# adds same-named classes in different submodules.
_CLASS_BY_TYPE = {
    "magnet.Cuboid": magpy.magnet.Cuboid,
    "magnet.Cylinder": magpy.magnet.Cylinder,
    "magnet.CylinderSegment": magpy.magnet.CylinderSegment,
    "current.Polyline": magpy.current.Polyline,
    "current.Circle": magpy.current.Circle,
    "Sensor": magpy.Sensor,
    "Collection": magpy.Collection,
}
_TYPE_BY_CLASS = {cls: tag for tag, cls in _CLASS_BY_TYPE.items()}


def to_json(*objs, **json_kwargs):
    """Serialize one or more magpylib objects to a JSON string.

    ``json_kwargs`` are forwarded to :func:`json.dumps` (e.g. ``indent=2``).
    """
    return json.dumps([_serialize_recursive(obj) for obj in objs], **json_kwargs)


def from_json(s):
    """Deserialize a JSON string produced by :func:`to_json`.

    Returns a list of magpylib objects.
    """
    return [_deserialize_recursive(obj) for obj in json.loads(s)]

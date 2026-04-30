from __future__ import annotations

import threading
import time
import warnings
from contextlib import contextmanager

import magpylib as magpy
from loguru import logger
from magpylib._src.obj_classes.class_BaseExcitations import BaseMagnet
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


def serialize_recursive(obj, parent="warn"):
    dd = {
        "id": id(obj),
        "type": obj.__class__.__name__,
        "position": {"value": obj.position.tolist(), "unit": "m"},
        "orientation": {
            "value": obj.orientation.as_matrix().tolist(),
            "type": "matrix",
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
        susceptibility = (
            getattr(obj, "susceptibility", None)
            if susceptibility is None
            else susceptibility
        )
        if susceptibility is not None:
            dd["susceptibility"] = {"value": susceptibility}
    if isinstance(obj, magpy.magnet.Cuboid):
        dd["dimension"] = {"value": obj.dimension.tolist(), "unit": "m"}
        susceptibility = getattr(obj, "susceptibility", None)
        susceptibility = (
            getattr(obj, "susceptibility", None)
            if susceptibility is None
            else susceptibility
        )
        if susceptibility is not None:
            dd["susceptibility"] = {"value": susceptibility}
    elif isinstance(obj, magpy.Sensor):
        dd["pixel"] = {"value": obj.pixel.tolist(), "unit": "m"}
    elif isinstance(obj, magpy.Collection):
        dd["children"] = [
            serialize_recursive(child, parent="ignore") for child in obj.children
        ]
    else:
        msg = "Only Cuboid supported"
        raise TypeError(msg)
    return dd


def deserialize_recursive(inp, ids=None):
    # constructor
    if ids is None:
        ids = {}
    typ = inp.get("type", None)
    is_coll = typ == "Collection"
    constr = object
    if typ == "Collection":
        constr = magpy.Collection
    elif typ == "Sensor":
        constr = magpy.Sensor
    elif getattr(magpy.magnet, typ, None) is not None:
        constr = getattr(magpy.magnet, typ)
    kw = {}
    # position
    kw["position"] = inp["position"]["value"]
    pos_unit = inp["position"]["unit"]
    if pos_unit != "m":
        msg = f"Position unit must be `m`, got {pos_unit!r}"
        raise ValueError(msg)

    # orientation
    orient = inp["orientation"]["value"]
    orient_typ = inp["orientation"]["type"]
    if orient_typ != "matrix":
        msg = f"Orientation type must be `matrix`, got {orient_typ!r}"
        raise ValueError(msg)
    kw["orientation"] = Rotation.from_matrix(orient)

    style = inp.get("style", None)
    if style is not None:
        kw["style"] = style

    if inp.get("parent", None) is not None:
        warnings.warn(f"object parent ({inp['parent']}) ignored", stacklevel=2)

    if issubclass(constr, BaseMagnet):
        kw["polarization"] = inp["polarization"]["value"]
        pol_unit = inp["polarization"]["unit"]
        if pol_unit != "T":
            msg = f"Polarization unit must be `T`, got {pol_unit!r}"
            raise ValueError(msg)
    if issubclass(constr, magpy.magnet.Cuboid):
        kw["dimension"] = inp["dimension"]["value"]
        dim_unit = inp["dimension"]["unit"]
        if dim_unit != "m":
            msg = f"Dimension unit must be `m`, got {dim_unit!r}"
            raise ValueError(msg)
    elif issubclass(constr, magpy.Sensor):
        kw["pixel"] = inp["pixel"]["value"]
        pix_unit = inp["pixel"]["unit"]
        if pix_unit != "m":
            msg = f"Pixel unit must be `m`, got {pix_unit!r}"
            raise ValueError(msg)
    elif not is_coll:
        msg = "Only Collection, Cuboid, Sensor supported"
        raise TypeError(msg)
    obj = constr(**kw)
    ids[inp["id"]] = obj
    if inp.get("susceptibility", None) is not None:
        obj.susceptibility = inp["susceptibility"]["value"]
    if is_coll:
        obj.add(*[deserialize_recursive(child, ids)[0] for child in inp["children"]])
    return obj, ids


def serialize_setup(*objs):
    res = []
    for obj in objs:
        res.append(serialize_recursive(obj))
    return res


def deserialize_setup(*objs, return_ids=False):
    res = []
    ids = {}
    for obj in objs:
        obj_list = []
        if not isinstance(obj, list | tuple):
            obj_list = [obj]
        for sub_obj in obj_list:
            r, i = deserialize_recursive(sub_obj)
            res.append(r)
            ids.update(i)
    if return_ids:
        res = res, ids
    return res

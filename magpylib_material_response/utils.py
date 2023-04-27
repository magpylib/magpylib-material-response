import threading
import time
import warnings
from contextlib import contextmanager

import magpylib as magpy
from loguru import logger
from magpylib._src.obj_classes.class_BaseExcitations import BaseMagnet
from scipy.spatial.transform import Rotation


class ElapsedTimeThread(threading.Thread):
    """ "Stoppable thread that logs the time elapsed"""

    def __init__(self, msg=None, min_log_time=1):
        super().__init__()
        self._stop_event = threading.Event()
        self.thread_start = time.time()
        self.msg = msg
        self.min_log_time = min_log_time
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
                logger.opt(colors=True).info(f"Start {self.msg}")
                self._msg_displayed = True
            # include a delay here so the thread doesn't uselessly thrash the CPU
            time.sleep(max(0.01, self.min_log_time / 5))


@contextmanager
def timelog(msg, min_log_time=1) -> float:
    """ "Measure and log time with loguru as context manager."""
    start = time.perf_counter()
    end = None
    thread_timer = ElapsedTimeThread(msg=msg, min_log_time=min_log_time)
    thread_timer.start()
    try:
        yield
        end = time.perf_counter() - start
    finally:
        thread_timer.stop()
        thread_timer.join()
        if end is None:
            logger.opt(colors=True).exception(f"{msg} failed")

    if end > min_log_time:
        logger.opt(colors=True).success(
            f"{msg} done" f"<green> 🕑 {round(end, 3)}sec</green>"
        )


def serialize_recursive(obj, parent="warn"):
    dd = {
        "id": id(obj),
        "type": obj.__class__.__name__,
        "position": {"value": obj.position.tolist(), "unit": "mm"},
        "orientation": {
            "value": obj.orientation.as_matrix().tolist(),
            "type": "matrix",
        },
    }
    if getattr(obj, "_style", None) is not None or obj._style_kwargs:
        dd["style"] = obj.style.as_dict()
    if parent == "warn" and obj.parent is not None:
        warnings.warn(f"object parent ({obj.parent}) not included in serialization")
    if isinstance(obj, BaseMagnet):
        dd["magnetization"] = {"value": obj.magnetization.tolist(), "unit": "mT"}
        xi = getattr(obj, "susceptibility", None)
        xi = getattr(obj, "xi", None) if xi is None else xi
        if xi is not None:
            dd["susceptibility"] = {"value": xi}
    if isinstance(obj, magpy.magnet.Cuboid):
        dd["dimension"] = {"value": obj.dimension.tolist(), "unit": "mm"}
        xi = getattr(obj, "susceptibility", None)
        xi = getattr(obj, "xi", None) if xi is None else xi
        if xi is not None:
            dd["susceptibility"] = {"value": xi}
    elif isinstance(obj, magpy.Sensor):
        dd["pixel"] = {"value": obj.pixel.tolist(), "unit": "mm"}
    elif isinstance(obj, magpy.Collection):
        dd["children"] = [
            serialize_recursive(child, parent="ignore") for child in obj.children
        ]
    else:
        raise TypeError("Only Cuboid supported")
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
    if pos_unit != "mm":
        raise ValueError(f"Position unit must be `mm`, got {pos_unit!r}")

    # orientation
    orient = inp["orientation"]["value"]
    orient_typ = inp["orientation"]["type"]
    if orient_typ != "matrix":
        raise ValueError(f"Orientation type must be `matrix`, got {orient_typ!r}")
    kw["orientation"] = Rotation.from_matrix(orient)

    style = inp.get("style", None)
    if style is not None:
        kw["style"] = style

    if inp.get("parent", None) is not None:
        warnings.warn(f"object parent ({inp['parent']}) ignored")

    if issubclass(constr, BaseMagnet):
        # magnetization
        kw["magnetization"] = inp["magnetization"]["value"]
        mag_unit = inp["magnetization"]["unit"]
        if mag_unit != "mT":
            raise ValueError(f"Magnetization unit must be `mT`, got {mag_unit!r}")
    if issubclass(constr, magpy.magnet.Cuboid):
        # dimension
        kw["dimension"] = inp["dimension"]["value"]
        dim_unit = inp["dimension"]["unit"]
        if dim_unit != "mm":
            raise ValueError(f"Dimension unit must be `mm`, got {dim_unit!r}")
    elif issubclass(constr, magpy.Sensor):
        kw["pixel"] = inp["pixel"]["value"]
        pix_unit = inp["pixel"]["unit"]
        if pix_unit != "mm":
            raise ValueError(f"Pixel unit must be `mm`, got {pix_unit!r}")
    elif not is_coll:
        raise TypeError("Only Collection, Cuboid, Sensor supported")
    obj = constr(**kw)
    ids[inp["id"]] = obj
    if inp.get("susceptibility", None) is not None:
        obj.xi = inp["susceptibility"]["value"]
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
        if not isinstance(obj, (list, tuple)):
            obj = [obj]
        for sub_obj in obj:
            r, i = deserialize_recursive(sub_obj)
            res.append(r)
            ids.update(i)
    if return_ids:
        res = res, ids
    return res

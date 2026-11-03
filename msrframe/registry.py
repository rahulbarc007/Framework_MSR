"""Plug-in registry.

Five kinds of interchangeable components are registered by name:

  ``surrogate``   scalar regressors  (gp, dnn, gbt, random_forest, pce, svr, ...)
  ``trajectory``  multi-time-point output models (pca, time_input, per_time)
  ``screening``   sensitivity / importance methods (morris, sobol_pce, random_forest, ...)
  ``optimizer``   box-constrained minimisers on the unit cube (differential_evolution, ...)
  ``sampler``     experimental designs (lhs, sobol, random, morris)

Register your own component with the decorator and refer to it from the YAML config::

    from msrframe import register
    from msrframe.models.scalar import Surrogate

    @register("surrogate", "my_model", description="my regressor")
    class MyModel(Surrogate):
        ...

and in the config:  ``keff: {models: [gp, my_model]}`` or ``plugins: [path/to/file.py]``.
"""
from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

KINDS = ("surrogate", "trajectory", "screening", "optimizer", "sampler")
_REG: dict[str, dict[str, dict]] = {k: {} for k in KINDS}


def register(kind: str, name: str, description: str = "", **meta):
    if kind not in _REG:
        raise KeyError(f"unknown plug-in kind {kind!r}; choose from {KINDS}")

    def deco(obj):
        _REG[kind][name] = dict(obj=obj, description=description or (obj.__doc__ or "").strip().split("\n")[0], **meta)
        obj.registry_name = name
        return obj

    return deco


def get(kind: str, name: str):
    try:
        return _REG[kind][name]["obj"]
    except KeyError:
        raise KeyError(f"no {kind} plug-in named {name!r}. Available: {sorted(_REG[kind])}") from None


def create(kind: str, spec, **extra):
    """Instantiate a plug-in from a name or a ``{name: ..., <params>}`` dict."""
    name, params = parse_spec(spec)
    params = {**params, **extra}
    return get(kind, name)(**params)


def parse_spec(spec):
    if isinstance(spec, str):
        return spec, {}
    if isinstance(spec, dict):
        spec = dict(spec)
        name = spec.pop("name", None) or spec.pop("method", None)
        if name is None:
            raise ValueError(f"plug-in spec needs a 'name': {spec}")
        spec.pop("label", None)
        return name, spec
    raise TypeError(f"bad plug-in spec {spec!r}")


def spec_label(spec) -> str:
    """Human label used in tables/plots: explicit 'label', else name[+base]."""
    if isinstance(spec, dict) and spec.get("label"):
        return spec["label"]
    name, params = parse_spec(spec)
    if "base" in params:
        b = params["base"]
        return f"{name}:{b if isinstance(b, str) else parse_spec(b)[0]}"
    if "surrogate" in params:
        s = params["surrogate"]
        return f"{name}:{s if isinstance(s, str) else parse_spec(s)[0]}"
    return name


def available(kind: str | None = None):
    if kind:
        return {n: v["description"] for n, v in sorted(_REG[kind].items())}
    return {k: available(k) for k in KINDS}


def load_plugin_file(path: str | Path):
    """Import a user python file (or dotted module) that calls :func:`register`."""
    p = Path(path)
    if p.suffix == ".py" and p.exists():
        spec = importlib.util.spec_from_file_location(p.stem, p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    return importlib.import_module(str(path))

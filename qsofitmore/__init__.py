#!/usr/bin/env python
import warnings

__all__ = []
from . import fitmodule
from .config import migration_config

# Friendly warning if user prefers kmpfit but kapteyn is not installed
if not migration_config.use_lmfit:
    kmp_avail = getattr(fitmodule, '_kmpfit', None) is not None
    if not kmp_avail:
        warnings.warn(
            "kapteyn.kmpfit is not installed; legacy kmpfit paths are unavailable. "
            "Either enable lmfit (migration_config.use_lmfit=True) or install Kapteyn: "
            "pip install 'cython<3.0' && pip install https://www.astro.rug.nl/software/kapteyn/kapteyn-3.4.tar.gz",
            RuntimeWarning,
        )

from . fitmodule import *  # noqa: E402,F401,F403
__all__ += fitmodule.__all__

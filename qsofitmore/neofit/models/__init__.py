"""Array model primitives for neofit."""

from .continuum import continuum, continuum_partials, normalized_coordinate
from .gaussian import gaussian, gaussian_partials

__all__ = [
    "continuum",
    "continuum_partials",
    "gaussian",
    "gaussian_partials",
    "normalized_coordinate",
]

"""Array-based fitting core for future qsofitmore workflows."""

from . import recipes
from .api import fit_line_complex, fit_local
from .config import GaussianComponent, IronTemplateConfig, LineComplexConfig, LocalFitConfig
from .host_workflow import NeoFitHostWorkflowResult, fit_with_optional_host_decomp
from .metadata import SpectrumMetadata, resolve_spectrum_metadata
from .plotting import plot_line_result, plot_local_result, save_local_window_plots
from .result import FitResult, LocalFitResult
from .spectrum import Spectrum
from .templates import IronTemplate, load_iron_template, list_iron_templates
from .warnings import NeoFitWarning

__all__ = [
    "FitResult",
    "GaussianComponent",
    "IronTemplate",
    "IronTemplateConfig",
    "LineComplexConfig",
    "LocalFitConfig",
    "LocalFitResult",
    "NeoFitWarning",
    "NeoFitHostWorkflowResult",
    "Spectrum",
    "SpectrumMetadata",
    "fit_line_complex",
    "fit_local",
    "fit_with_optional_host_decomp",
    "list_iron_templates",
    "load_iron_template",
    "plot_line_result",
    "plot_local_result",
    "recipes",
    "resolve_spectrum_metadata",
    "save_local_window_plots",
]

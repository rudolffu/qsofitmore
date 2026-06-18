"""Array-based fitting core for future qsofitmore workflows."""

from . import recipes
from .api import fit_line_complex, fit_local
from .config import (
    BalmerContinuumConfig,
    BalmerSeriesConfig,
    GaussianComponent,
    GlobalContinuumConfig,
    HalphaComplexConfig,
    HbetaComplexConfig,
    IronTemplateConfig,
    LineComplexConfig,
    LocalFitConfig,
    LorentzianComponent,
    MgIIComplexConfig,
    PowerLawConfig,
    UncertaintyConfig,
)
from .global_fit import (
    balmer_continuum_basis,
    fit_global_continuum,
    fit_global_hbeta,
    fit_global_lines,
    fit_halpha_complex,
    fit_hbeta_complex,
    fit_mgii_complex,
)
from .global_io import (
    GlobalQAPlotConfig,
    write_global_hbeta_products,
    write_global_line_products,
)
from .global_result import (
    EmissionComplexResult,
    GlobalContinuumResult,
    HbetaComplexResult,
    NeoFitWorkflowResult,
)
from .host_workflow import (
    NeoFitHostWorkflowResult,
    fit_global_hbeta_workflow,
    fit_global_lines_workflow,
    fit_with_optional_host_decomp,
)
from .metadata import SpectrumMetadata, resolve_spectrum_metadata
from .plotting import plot_line_result, plot_local_result, save_local_window_plots
from .result import FitResult, LocalFitResult
from .spectrum import Spectrum
from .templates import (
    BalmerSeriesTemplate,
    IronTemplate,
    list_balmer_templates,
    list_iron_templates,
    load_balmer_template,
    load_iron_template,
)
from .warnings import NeoFitWarning

__all__ = [
    "BalmerContinuumConfig",
    "BalmerSeriesConfig",
    "BalmerSeriesTemplate",
    "FitResult",
    "EmissionComplexResult",
    "GaussianComponent",
    "GlobalContinuumConfig",
    "GlobalContinuumResult",
    "GlobalQAPlotConfig",
    "HalphaComplexConfig",
    "HbetaComplexConfig",
    "HbetaComplexResult",
    "IronTemplate",
    "IronTemplateConfig",
    "LineComplexConfig",
    "LocalFitConfig",
    "LocalFitResult",
    "LorentzianComponent",
    "MgIIComplexConfig",
    "NeoFitWarning",
    "NeoFitHostWorkflowResult",
    "NeoFitWorkflowResult",
    "PowerLawConfig",
    "Spectrum",
    "SpectrumMetadata",
    "UncertaintyConfig",
    "balmer_continuum_basis",
    "fit_global_continuum",
    "fit_global_hbeta",
    "fit_global_hbeta_workflow",
    "fit_global_lines",
    "fit_global_lines_workflow",
    "fit_halpha_complex",
    "fit_hbeta_complex",
    "fit_mgii_complex",
    "fit_line_complex",
    "fit_local",
    "fit_with_optional_host_decomp",
    "list_balmer_templates",
    "list_iron_templates",
    "load_balmer_template",
    "load_iron_template",
    "plot_line_result",
    "plot_local_result",
    "recipes",
    "resolve_spectrum_metadata",
    "save_local_window_plots",
    "write_global_hbeta_products",
    "write_global_line_products",
]

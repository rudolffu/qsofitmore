"""Template loading and normalization helpers for neofit."""

from .balmer import (
    BalmerSeriesTemplate,
    BalmerTemplateError,
    evaluate_balmer_series,
    evaluate_balmer_series_with_derivative,
    list_balmer_templates,
    load_balmer_template,
)
from .iron import (
    IronTemplate,
    IronTemplateError,
    PreparedIronTemplate,
    evaluate_iron_basis,
    evaluate_iron_basis_with_derivative,
    prepare_iron_template,
)
from .registry import list_iron_templates, load_iron_template, resolve_iron_template_name

__all__ = [
    "BalmerSeriesTemplate",
    "BalmerTemplateError",
    "IronTemplate",
    "IronTemplateError",
    "PreparedIronTemplate",
    "evaluate_balmer_series",
    "evaluate_balmer_series_with_derivative",
    "evaluate_iron_basis",
    "evaluate_iron_basis_with_derivative",
    "list_balmer_templates",
    "list_iron_templates",
    "load_balmer_template",
    "load_iron_template",
    "prepare_iron_template",
    "resolve_iron_template_name",
]

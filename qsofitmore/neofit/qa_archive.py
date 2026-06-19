"""Post-fit QA rendering from Parquet model archives."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from .global_io import (
    GlobalQAPlotConfig,
    _plot_paths,
    _plot_qa,
    _qa_object_name,
    _save_figure,
    _normalized_file_label,
)
from .run_store import RunStore, load_model, open_run


def render_qa(
    run: str | RunStore,
    *,
    object_ids: Optional[Sequence[str]] = None,
    warning_codes: Optional[Sequence[str]] = None,
    query: Optional[str] = None,
    sample: Optional[int] = None,
    random_seed: int = 12345,
    include_failed: bool = False,
    plot_config: Optional[GlobalQAPlotConfig] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Dict[str, str]]:
    """Render main QA figures without refitting archived objects."""

    store = open_run(run) if isinstance(run, str) else run
    objects = store.read_table("objects").to_pandas()
    if query:
        objects = objects.query(query)
    if object_ids is not None:
        requested = set(map(str, object_ids))
        objects = objects[
            objects["object_id"].astype(str).isin(requested)
            | objects["object_key"].astype(str).isin(requested)
        ]
    if warning_codes:
        requested_warnings = set(map(str, warning_codes))
        objects = objects[
            objects["warning_codes"].map(
                lambda values: bool(requested_warnings.intersection(values or ()))
            )
        ]
    if sample is not None and len(objects) > int(sample):
        rng = np.random.default_rng(random_seed)
        indices = np.sort(
            rng.choice(len(objects), size=int(sample), replace=False)
        )
        objects = objects.iloc[indices]
    config = plot_config or GlobalQAPlotConfig()
    destination = Path(output_dir).expanduser() if output_dir else store.path / "qa"
    destination.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, Dict[str, str]] = {}
    for row in objects.sort_values("object_key").to_dict("records"):
        result = load_model(store, row["object_key"])
        if config.object_name in (None, ""):
            object_config = GlobalQAPlotConfig(
                **{
                    **asdict(config),
                    "object_name": row["object_id"],
                }
            )
        else:
            object_config = config
        label = _qa_object_name(result, object_config)
        saved = _plot_qa(
            result,
            _plot_paths(destination, f"main_qa_{label}", object_config),
            object_config,
        )
        primary = saved.get("png", next(iter(saved.values())))
        outputs[str(row["object_id"])] = {
            **saved,
            **{f"main_qa_{key}": value for key, value in saved.items()},
            "main_qa": primary,
            "global_plot": primary,
            "qa_plot": primary,
        }
    if include_failed:
        import matplotlib.pyplot as plt

        failures = store.read_table("failures").to_pandas()
        if object_ids is not None:
            requested = set(map(str, object_ids))
            failures = failures[
                failures["object_id"].astype(str).isin(requested)
                | failures["object_key"].astype(str).isin(requested)
            ]
        for row in failures.sort_values("object_key").to_dict("records"):
            label = _normalized_file_label(
                row["object_id"] or row["object_key"]
            )
            paths = _plot_paths(destination, f"failed_qa_{label}", config)
            fig, axis = plt.subplots(
                figsize=(config.figure_width, 3.2),
                constrained_layout=True,
            )
            axis.axis("off")
            axis.text(
                0.02,
                0.95,
                f"Fit failed — {row['object_id'] or row['object_key']}",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=13,
            )
            axis.text(
                0.02,
                0.78,
                f"{row['exception_type']}: {row['message']}",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=10,
                wrap=True,
            )
            outputs[str(row["object_id"] or row["object_key"])] = _save_figure(
                fig, paths
            )
            plt.close(fig)
    return outputs

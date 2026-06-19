"""Parquet-backed immutable run bundles for neofit results."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from contextlib import contextmanager
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Callable, Dict, Mapping, Optional, Sequence
from uuid import uuid4

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq

from .global_result import (
    EmissionComplexResult,
    GlobalContinuumResult,
    NeoFitWorkflowResult,
)
from .metadata import SpectrumMetadata
from .spectrum import Spectrum
from .warnings import NeoFitWarning


SCHEMA_VERSION = "1"
TABLE_NAMES = (
    "inputs",
    "objects",
    "measurements",
    "warnings",
    "models",
    "failures",
    "derived",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if callable(value):
        return getattr(value, "__qualname__", repr(value))
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def configuration_hash(configuration: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 hash for a run configuration."""

    payload = json.dumps(
        _jsonable(configuration),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _package_version() -> str:
    try:
        return version("qsofitmore")
    except PackageNotFoundError:
        return "unknown"


def _git_commit(root: Path) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _key_values(mapping: Mapping[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "key": str(key),
            "value": json.dumps(_jsonable(value), sort_keys=True, allow_nan=True),
        }
        for key, value in sorted(mapping.items(), key=lambda item: str(item[0]))
    ]


def _from_key_values(items: Optional[Sequence[Mapping[str, str]]]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for item in items or ():
        try:
            output[str(item["key"])] = json.loads(item["value"])
        except Exception:
            output[str(item["key"])] = item.get("value")
    return output


KEY_VALUE_TYPE = pa.list_(
    pa.struct([pa.field("key", pa.string()), pa.field("value", pa.string())])
)
COMPONENT_TYPE = pa.list_(
    pa.struct(
        [
            pa.field("section", pa.string()),
            pa.field("recipe_id", pa.string()),
            pa.field("name", pa.string()),
            pa.field("role", pa.string()),
            pa.field("values", pa.list_(pa.float64())),
        ]
    )
)
COMPLEX_TYPE = pa.list_(
    pa.struct(
        [
            pa.field("recipe_id", pa.string()),
            pa.field("success", pa.bool_()),
            pa.field("status", pa.int32()),
            pa.field("message", pa.string()),
            pa.field("selected_model", pa.string()),
            pa.field("chi2", pa.float64()),
            pa.field("dof", pa.int64()),
            pa.field("reduced_chi2", pa.float64()),
            pa.field("bic", pa.float64()),
            pa.field("model", pa.list_(pa.float64())),
            pa.field("flux_continuum_subtracted", pa.list_(pa.float64())),
            pa.field("fit_mask", pa.list_(pa.bool_())),
        ]
    )
)

SCHEMAS = {
    "inputs": pa.schema(
        [
            pa.field("run_id", pa.string()),
            pa.field("object_key", pa.string()),
            pa.field("object_id", pa.string()),
            pa.field("source", pa.string()),
            pa.field("row_index", pa.int64()),
            pa.field("reader", pa.string()),
            pa.field("redshift", pa.float64()),
            pa.field("metadata", KEY_VALUE_TYPE),
        ]
    ),
    "objects": pa.schema(
        [
            pa.field("run_id", pa.string()),
            pa.field("object_key", pa.string()),
            pa.field("object_id", pa.string()),
            pa.field("redshift", pa.float64()),
            pa.field("ra", pa.float64()),
            pa.field("dec", pa.float64()),
            pa.field("continuum_success", pa.bool_()),
            pa.field("continuum_reduced_chi2", pa.float64()),
            pa.field("host_decomp_enabled", pa.bool_()),
            pa.field("complex_statuses", KEY_VALUE_TYPE),
            pa.field("warning_codes", pa.list_(pa.string())),
            pa.field("metadata", KEY_VALUE_TYPE),
            pa.field("completed_at", pa.string()),
        ]
    ),
    "measurements": pa.schema(
        [
            pa.field("run_id", pa.string()),
            pa.field("object_key", pa.string()),
            pa.field("object_id", pa.string()),
            pa.field("section", pa.string()),
            pa.field("recipe_id", pa.string()),
            pa.field("feature_id", pa.string()),
            pa.field("role", pa.string()),
            pa.field("quantity", pa.string()),
            pa.field("value", pa.float64()),
            pa.field("error", pa.float64()),
            pa.field("unit", pa.string()),
            pa.field("method", pa.string()),
            pa.field("metadata", KEY_VALUE_TYPE),
        ]
    ),
    "warnings": pa.schema(
        [
            pa.field("run_id", pa.string()),
            pa.field("object_key", pa.string()),
            pa.field("object_id", pa.string()),
            pa.field("section", pa.string()),
            pa.field("recipe_id", pa.string()),
            pa.field("code", pa.string()),
            pa.field("severity", pa.string()),
            pa.field("message", pa.string()),
            pa.field("context", KEY_VALUE_TYPE),
        ]
    ),
    "models": pa.schema(
        [
            pa.field("run_id", pa.string()),
            pa.field("object_key", pa.string()),
            pa.field("object_id", pa.string()),
            pa.field("redshift", pa.float64()),
            pa.field("wave_obs", pa.list_(pa.float64())),
            pa.field("flux", pa.list_(pa.float64())),
            pa.field("error", pa.list_(pa.float64())),
            pa.field("input_mask", pa.list_(pa.bool_())),
            pa.field("total_flux", pa.list_(pa.float64())),
            pa.field("host_model", pa.list_(pa.float64())),
            pa.field("continuum_model", pa.list_(pa.float64())),
            pa.field("continuum_fit_mask", pa.list_(pa.bool_())),
            pa.field("continuum_clip_mask", pa.list_(pa.bool_())),
            pa.field("full_model", pa.list_(pa.float64())),
            pa.field("components", COMPONENT_TYPE),
            pa.field("complexes", COMPLEX_TYPE),
            pa.field("spectrum_metadata", KEY_VALUE_TYPE),
            pa.field("workflow_metadata", KEY_VALUE_TYPE),
        ]
    ),
    "failures": pa.schema(
        [
            pa.field("run_id", pa.string()),
            pa.field("object_key", pa.string()),
            pa.field("object_id", pa.string()),
            pa.field("source", pa.string()),
            pa.field("row_index", pa.int64()),
            pa.field("exception_type", pa.string()),
            pa.field("message", pa.string()),
            pa.field("traceback", pa.string()),
            pa.field("failed_at", pa.string()),
            pa.field("metadata", KEY_VALUE_TYPE),
        ]
    ),
    "derived": pa.schema(
        [
            pa.field("run_id", pa.string()),
            pa.field("object_key", pa.string()),
            pa.field("object_id", pa.string()),
            pa.field("quantity", pa.string()),
            pa.field("calibration_id", pa.string()),
            pa.field("value", pa.float64()),
            pa.field("statistical_error", pa.float64()),
            pa.field("intrinsic_scatter", pa.float64()),
            pa.field("total_error", pa.float64()),
            pa.field("unit", pa.string()),
            pa.field("metadata", KEY_VALUE_TYPE),
        ]
    ),
}


def _float(value: Any) -> Optional[float]:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    return output


def _feature_and_role(name: str) -> tuple[Optional[str], Optional[str]]:
    lowered = name.lower()
    roles = ("very_broad", "broad", "narrow", "wing", "blend")
    for role in roles:
        token = f"_{role}_"
        if token in lowered:
            index = lowered.index(token)
            return name[:index], role
    return None, None


def _measurement_rows(
    result: NeoFitWorkflowResult,
    run_id: str,
    object_key: str,
    object_id: str,
) -> list[dict[str, Any]]:
    rows = []

    def add(
        section: str,
        recipe_id: Optional[str],
        values: Mapping[str, Any],
        errors: Mapping[str, Any],
        method: str,
    ) -> None:
        for quantity, value in values.items():
            numeric = _float(value)
            if numeric is None:
                continue
            feature_id, role = _feature_and_role(str(quantity))
            rows.append(
                {
                    "run_id": run_id,
                    "object_key": object_key,
                    "object_id": object_id,
                    "section": section,
                    "recipe_id": recipe_id,
                    "feature_id": feature_id,
                    "role": role,
                    "quantity": str(quantity),
                    "value": numeric,
                    "error": _float(errors.get(quantity)),
                    "unit": None,
                    "method": method,
                    "metadata": [],
                }
            )

    add(
        "continuum_parameter",
        None,
        result.continuum.param_values,
        result.continuum.param_errors,
        "covariance",
    )
    for quantity in (
        "balmer_series_implied_hbeta_flux_input",
        "balmer_series_implied_hbeta_flux_cgs",
        "balmer_series_fwhm_kms",
    ):
        if quantity in result.continuum.metadata:
            add(
                "continuum_metric",
                None,
                {quantity: result.continuum.metadata[quantity]},
                {},
                "fit_metadata",
            )
    for recipe_id, fit in result.line_complexes.items():
        add(
            "complex_parameter",
            recipe_id,
            fit.param_values,
            fit.param_errors,
            "covariance",
        )
        add(
            "complex_metric",
            recipe_id,
            fit.metrics,
            fit.metric_errors,
            "covariance",
        )
    for quantity, value in result.metadata.get("continuum_samples", {}).items():
        add("continuum_sample", None, {quantity: value}, {}, "interpolation")
    return rows


def _warning_rows(
    result: NeoFitWorkflowResult,
    run_id: str,
    object_key: str,
    object_id: str,
) -> list[dict[str, Any]]:
    rows = []

    def add(section, recipe_id, warnings):
        for warning in warnings:
            rows.append(
                {
                    "run_id": run_id,
                    "object_key": object_key,
                    "object_id": object_id,
                    "section": section,
                    "recipe_id": recipe_id,
                    "code": warning.code,
                    "severity": warning.severity,
                    "message": warning.message,
                    "context": _key_values(warning.context),
                }
            )

    add("workflow", None, result.warnings)
    add("continuum", None, result.continuum.warnings)
    for recipe_id, fit in result.line_complexes.items():
        add("complex", recipe_id, fit.warnings)
    for message in result.host_warnings:
        rows.append(
            {
                "run_id": run_id,
                "object_key": object_key,
                "object_id": object_id,
                "section": "host",
                "recipe_id": None,
                "code": "host_warning",
                "severity": "warning",
                "message": str(message),
                "context": [],
            }
        )
    return rows


def _component_role(name: str) -> str:
    lowered = name.lower()
    for role in ("very_broad", "broad", "narrow", "wing", "blend"):
        if role in lowered:
            return role
    return "continuum"


def _model_row(
    result: NeoFitWorkflowResult,
    run_id: str,
    object_key: str,
    object_id: str,
) -> dict[str, Any]:
    components = [
        {
            "section": "continuum",
            "recipe_id": None,
            "name": name,
            "role": "continuum",
            "values": np.asarray(values, dtype=float).tolist(),
        }
        for name, values in result.continuum.component_models.items()
    ]
    complexes = []
    line_sum = np.zeros_like(result.continuum.model)
    for recipe_id, fit in result.line_complexes.items():
        if fit.success:
            line_sum += fit.model
        components.extend(
            {
                "section": "complex",
                "recipe_id": recipe_id,
                "name": name,
                "role": _component_role(name),
                "values": np.asarray(values, dtype=float).tolist(),
            }
            for name, values in fit.component_models.items()
        )
        complexes.append(
            {
                "recipe_id": recipe_id,
                "success": bool(fit.success),
                "status": int(fit.status),
                "message": str(fit.message),
                "selected_model": str(fit.selected_model),
                "chi2": float(fit.chi2),
                "dof": int(fit.dof),
                "reduced_chi2": float(fit.reduced_chi2),
                "bic": float(fit.bic),
                "model": np.asarray(fit.model, dtype=float).tolist(),
                "flux_continuum_subtracted": np.asarray(
                    fit.flux_continuum_subtracted, dtype=float
                ).tolist(),
                "fit_mask": np.asarray(fit.fit_mask, dtype=bool).tolist(),
            }
        )
    return {
        "run_id": run_id,
        "object_key": object_key,
        "object_id": object_id,
        "redshift": float(result.spectrum.z),
        "wave_obs": np.asarray(result.spectrum.wave_obs, dtype=float).tolist(),
        "flux": np.asarray(result.spectrum.flux, dtype=float).tolist(),
        "error": np.asarray(result.spectrum.err, dtype=float).tolist(),
        "input_mask": (
            np.asarray(result.spectrum.mask, dtype=bool).tolist()
            if result.spectrum.mask is not None else None
        ),
        "total_flux": (
            np.asarray(result.total_spectrum.flux, dtype=float).tolist()
            if result.total_spectrum is not None else None
        ),
        "host_model": (
            np.asarray(result.host_model_on_quasar_grid, dtype=float).tolist()
            if result.host_model_on_quasar_grid is not None else None
        ),
        "continuum_model": np.asarray(
            result.continuum.model, dtype=float
        ).tolist(),
        "continuum_fit_mask": np.asarray(
            result.continuum.fit_mask, dtype=bool
        ).tolist(),
        "continuum_clip_mask": np.asarray(
            result.continuum.clip_mask, dtype=bool
        ).tolist(),
        "full_model": np.asarray(
            result.continuum.model + line_sum, dtype=float
        ).tolist(),
        "components": components,
        "complexes": complexes,
        "spectrum_metadata": _key_values(result.spectrum.metadata.to_dict()),
        "workflow_metadata": _key_values(result.metadata),
    }


def workflow_payload(
    result: NeoFitWorkflowResult,
    *,
    run_id: str,
    object_key: str,
    object_id: str,
    input_record: Mapping[str, Any],
) -> Dict[str, list[dict[str, Any]]]:
    """Serialize one workflow into all authoritative run tables."""

    metadata = dict(result.metadata)
    return {
        "inputs": [
            {
                "run_id": run_id,
                "object_key": object_key,
                "object_id": object_id,
                "source": str(input_record.get("source", "")),
                "row_index": input_record.get("row_index"),
                "reader": str(input_record.get("reader", "auto")),
                "redshift": float(result.spectrum.z),
                "metadata": _key_values(input_record.get("metadata", {})),
            }
        ],
        "objects": [
            {
                "run_id": run_id,
                "object_key": object_key,
                "object_id": object_id,
                "redshift": float(result.spectrum.z),
                "ra": _float(metadata.get("ra")),
                "dec": _float(metadata.get("dec")),
                "continuum_success": bool(result.continuum_success),
                "continuum_reduced_chi2": float(
                    result.continuum.reduced_chi2
                ),
                "host_decomp_enabled": bool(result.host_decomp_enabled),
                "complex_statuses": _key_values(result.complex_statuses),
                "warning_codes": list(result.warning_codes()),
                "metadata": _key_values(metadata),
                "completed_at": _now(),
            }
        ],
        "measurements": _measurement_rows(
            result, run_id, object_key, object_id
        ),
        "warnings": _warning_rows(result, run_id, object_key, object_id),
        "models": [_model_row(result, run_id, object_key, object_id)],
        "failures": [],
        "derived": [],
    }


def _empty_table(name: str) -> pa.Table:
    return pa.Table.from_pylist([], schema=SCHEMAS[name])


@dataclass
class RunStore:
    """Opened immutable run bundle."""

    path: Path
    manifest: Dict[str, Any] = field(default_factory=dict)

    @property
    def run_id(self) -> str:
        return str(self.manifest["run_id"])

    @property
    def configuration_hash(self) -> str:
        return str(self.manifest["configuration_hash"])

    @classmethod
    def create(
        cls,
        path: str,
        *,
        configuration: Mapping[str, Any],
        run_id: Optional[str] = None,
        resume: bool = True,
    ) -> "RunStore":
        root = Path(path).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        manifest_path = root / "manifest.json"
        config_hash = configuration_hash(configuration)
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("configuration_hash") != config_hash:
                raise ValueError(
                    "Run configuration does not match the immutable manifest. "
                    "Choose a new run directory or run_id."
                )
            if not resume:
                raise FileExistsError(f"Run already exists: {root}")
            store = cls(root, manifest)
            store._ensure_directories()
            return store
        actual_run_id = run_id or (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-"
            + config_hash[:10]
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "run_id": actual_run_id,
            "configuration_hash": config_hash,
            "configuration": _jsonable(configuration),
            "cosmology": _jsonable(configuration.get("cosmology")),
            "units": {
                "wavelength": "vacuum Angstrom",
                "velocity": "km/s",
                "model_arrays": "input flux-density units",
            },
            "created_at": _now(),
            "updated_at": _now(),
            "package_version": _package_version(),
            "git_commit": _git_commit(Path(__file__).resolve().parents[2]),
            "status": "active",
            "tables": list(TABLE_NAMES),
            "completed_objects": 0,
            "failed_objects": 0,
        }
        store = cls(root, manifest)
        store._ensure_directories()
        store._write_manifest()
        return store

    @classmethod
    def open(cls, path: str) -> "RunStore":
        root = Path(path).expanduser()
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        return cls(root, manifest)

    def _ensure_directories(self) -> None:
        for name in TABLE_NAMES:
            (self.path / name).mkdir(parents=True, exist_ok=True)
        (self.path / "compact").mkdir(exist_ok=True)
        (self.path / "qa").mkdir(exist_ok=True)
        (self.path / "staging").mkdir(exist_ok=True)

    def _write_manifest(self) -> None:
        with self._manifest_lock():
            manifest_path = self.path / "manifest.json"
            if manifest_path.exists():
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
                existing.update(self.manifest)
                self.manifest = existing
            self.manifest["updated_at"] = _now()
            if (self.path / "objects").exists():
                self.manifest["completed_objects"] = len(self.completed_keys())
                self.manifest["failed_objects"] = len(self.failed_keys())
                self.manifest["shard_state"] = {
                    name: len(tuple((self.path / name).glob("*.parquet")))
                    for name in TABLE_NAMES
                }
            temporary = self.path / f"manifest.{uuid4().hex}.tmp"
            temporary.write_text(
                json.dumps(self.manifest, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            os.replace(temporary, manifest_path)

    @contextmanager
    def _manifest_lock(self):
        lock_path = self.path / ".manifest.lock"
        handle = lock_path.open("a+")
        try:
            try:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            except ImportError:
                pass
            yield
        finally:
            try:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except ImportError:
                pass
            handle.close()

    def completed_keys(self) -> set[str]:
        table = self.read_table("objects")
        return set(table.column("object_key").to_pylist()) if table.num_rows else set()

    def failed_keys(self) -> set[str]:
        table = self.read_table("failures")
        return set(table.column("object_key").to_pylist()) if table.num_rows else set()

    def clear_failure(self, object_key: str) -> None:
        digest = hashlib.sha256(object_key.encode("utf-8")).hexdigest()[:20]
        path = self.path / "failures" / f"part-{digest}.parquet"
        if path.exists():
            path.unlink()

    def stage_payload(
        self,
        payload: Mapping[str, Sequence[Mapping[str, Any]]],
        *,
        namespace: Optional[str] = None,
    ) -> Path:
        """Write one private, collision-free staging directory."""

        namespace = namespace or f"{os.getpid()}-{uuid4().hex}"
        staging = self.path / "staging" / namespace
        staging.mkdir(parents=True, exist_ok=False)
        object_keys = {
            str(row["object_key"])
            for rows in payload.values()
            for row in rows
            if row.get("object_key") is not None
        }
        checksums = {}
        for name in TABLE_NAMES:
            rows = list(payload.get(name, ()))
            if not rows:
                continue
            table = pa.Table.from_pylist(rows, schema=SCHEMAS[name])
            output = staging / f"{name}.parquet"
            pq.write_table(table, output, compression="zstd")
            checksums[output.name] = hashlib.sha256(output.read_bytes()).hexdigest()
        (staging / "staging.json").write_text(
            json.dumps(
                {
                    "object_keys": sorted(object_keys),
                    "replace_tables": sorted(
                        name for name in payload if name in TABLE_NAMES
                    ),
                    "checksums": checksums,
                }
            ),
            encoding="utf-8",
        )
        return staging

    def promote(self, staging: str | Path) -> Dict[str, str]:
        """Validate and atomically promote a worker staging directory."""

        source = Path(staging)
        promoted: Dict[str, str] = {}
        staging_metadata = json.loads(
            (source / "staging.json").read_text(encoding="utf-8")
        )
        object_keys = staging_metadata.get("object_keys", [])
        replace_tables = set(staging_metadata.get("replace_tables", ()))
        checksums = staging_metadata.get("checksums", {})
        for filename, expected in checksums.items():
            actual = hashlib.sha256((source / filename).read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"Staged shard checksum mismatch: {filename}")
        if len(object_keys) == 1:
            digest = hashlib.sha256(
                object_keys[0].encode("utf-8")
            ).hexdigest()[:20]
            for name in replace_tables:
                old_path = self.path / name / f"part-{digest}.parquet"
                if old_path.exists():
                    old_path.unlink()
        for file_path in sorted(source.glob("*.parquet")):
            name = file_path.stem
            if name not in SCHEMAS:
                raise ValueError(f"Unknown staged table: {name}")
            table = pq.read_table(file_path)
            if not table.schema.equals(SCHEMAS[name], check_metadata=False):
                table = table.cast(SCHEMAS[name])
                pq.write_table(table, file_path, compression="zstd")
            object_key = (
                table.column("object_key")[0].as_py()
                if table.num_rows else uuid4().hex
            )
            digest = hashlib.sha256(object_key.encode("utf-8")).hexdigest()[:20]
            destination = self.path / name / f"part-{digest}.parquet"
            os.replace(file_path, destination)
            promoted[name] = str(destination)
        shutil.rmtree(source, ignore_errors=True)
        self._write_manifest()
        return promoted

    def write_payload(
        self,
        payload: Mapping[str, Sequence[Mapping[str, Any]]],
    ) -> Dict[str, str]:
        return self.promote(self.stage_payload(payload))

    def read_table(
        self,
        name: str,
        *,
        columns: Optional[Sequence[str]] = None,
        filter_expression: Any = None,
    ) -> pa.Table:
        if name not in SCHEMAS:
            raise ValueError(f"Unknown run table: {name!r}")
        files = sorted((self.path / name).glob("*.parquet"))
        if not files:
            table = _empty_table(name)
            return table.select(columns) if columns else table
        dataset = pads.dataset([str(path) for path in files], format="parquet")
        return dataset.to_table(columns=columns, filter=filter_expression)

    def object_row(
        self,
        identifier: str,
        *,
        table_name: str = "models",
    ) -> Mapping[str, Any]:
        table = self.read_table(table_name)
        matches = [
            row
            for row in table.to_pylist()
            if row["object_key"] == identifier or row["object_id"] == identifier
        ]
        if not matches:
            raise KeyError(f"Object not found in {table_name}: {identifier!r}")
        if len(matches) > 1:
            raise ValueError(
                f"Object identifier is ambiguous; use object_key: {identifier!r}"
            )
        return matches[0]


def open_run(path: str) -> RunStore:
    """Open an existing run bundle."""

    return RunStore.open(path)


def _measurement_maps(
    store: RunStore,
    object_key: str,
    section: str,
    recipe_id: Optional[str],
) -> tuple[Dict[str, float], Dict[str, float]]:
    rows = [
        row
        for row in store.read_table("measurements").to_pylist()
        if row["object_key"] == object_key
        and row["section"] == section
        and row["recipe_id"] == recipe_id
    ]
    values = {row["quantity"]: row["value"] for row in rows}
    errors = {row["quantity"]: row["error"] for row in rows}
    return values, errors


def load_model(run: str | RunStore, identifier: str) -> NeoFitWorkflowResult:
    """Reconstruct a workflow result from the Parquet model archive."""

    store = open_run(run) if isinstance(run, str) else run
    row = store.object_row(identifier, table_name="models")
    object_row = store.object_row(row["object_key"], table_name="objects")
    spectrum_metadata = SpectrumMetadata(**_from_key_values(row["spectrum_metadata"]))
    spectrum = Spectrum.from_arrays(
        np.asarray(row["wave_obs"], dtype=float),
        np.asarray(row["flux"], dtype=float),
        err=np.asarray(row["error"], dtype=float),
        z=float(row["redshift"]),
        mask=(
            np.asarray(row["input_mask"], dtype=bool)
            if row["input_mask"] is not None else None
        ),
        metadata=spectrum_metadata,
    )
    total_spectrum = None
    if row["total_flux"] is not None:
        total_spectrum = Spectrum.from_arrays(
            np.asarray(row["wave_obs"], dtype=float),
            np.asarray(row["total_flux"], dtype=float),
            err=np.asarray(row["error"], dtype=float),
            z=float(row["redshift"]),
            mask=(
                np.asarray(row["input_mask"], dtype=bool)
                if row["input_mask"] is not None else None
            ),
            metadata=spectrum_metadata,
        )
    continuum_components = {
        item["name"]: np.asarray(item["values"], dtype=float)
        for item in row["components"]
        if item["section"] == "continuum"
    }
    continuum_values, continuum_errors = _measurement_maps(
        store, row["object_key"], "continuum_parameter", None
    )
    workflow_metadata = _from_key_values(row["workflow_metadata"])
    continuum = GlobalContinuumResult(
        success=bool(object_row["continuum_success"]),
        status=1 if object_row["continuum_success"] else -1,
        message="Loaded from Parquet model archive.",
        param_values=continuum_values,
        param_errors=continuum_errors,
        covariance=None,
        chi2=np.nan,
        dof=0,
        reduced_chi2=float(object_row["continuum_reduced_chi2"]),
        wave_rest=spectrum.wave_rest.copy(),
        model=np.asarray(row["continuum_model"], dtype=float),
        component_models=continuum_components,
        fit_mask=np.asarray(row["continuum_fit_mask"], dtype=bool),
        clip_mask=np.asarray(row["continuum_clip_mask"], dtype=bool),
        metadata=workflow_metadata,
    )
    complex_components: Dict[str, Dict[str, np.ndarray]] = {}
    for component in row["components"]:
        if component["section"] == "complex":
            complex_components.setdefault(component["recipe_id"], {})[
                component["name"]
            ] = np.asarray(component["values"], dtype=float)
    complexes: Dict[str, EmissionComplexResult] = {}
    for item in row["complexes"]:
        recipe_id = item["recipe_id"]
        parameters, parameter_errors = _measurement_maps(
            store, row["object_key"], "complex_parameter", recipe_id
        )
        metrics, metric_errors = _measurement_maps(
            store, row["object_key"], "complex_metric", recipe_id
        )
        complexes[recipe_id] = EmissionComplexResult(
            success=bool(item["success"]),
            status=int(item["status"]),
            message=str(item["message"]),
            selected_model=str(item["selected_model"]),
            param_values=parameters,
            param_errors=parameter_errors,
            covariance=None,
            metrics=metrics,
            metric_errors=metric_errors,
            chi2=float(item["chi2"]),
            dof=int(item["dof"]),
            reduced_chi2=float(item["reduced_chi2"]),
            bic=float(item["bic"]),
            wave_rest=spectrum.wave_rest.copy(),
            flux_continuum_subtracted=np.asarray(
                item["flux_continuum_subtracted"], dtype=float
            ),
            err=spectrum.err.copy(),
            model=np.asarray(item["model"], dtype=float),
            component_models=complex_components.get(recipe_id, {}),
            fit_mask=np.asarray(item["fit_mask"], dtype=bool),
            metadata={"recipe_id": recipe_id},
        )
    warnings = [
        NeoFitWarning(
            code=item["code"],
            message=item["message"],
            severity=item["severity"],
            context=_from_key_values(item["context"]),
        )
        for item in store.read_table("warnings").to_pylist()
        if item["object_key"] == row["object_key"] and item["section"] == "workflow"
    ]
    workflow = NeoFitWorkflowResult(
        spectrum=spectrum,
        continuum_initial=continuum,
        continuum=continuum,
        hbeta=complexes.get("hbeta_oiii"),
        hbeta_initial=complexes.get("hbeta_oiii"),
        mgii=complexes.get("mgii"),
        halpha=complexes.get("halpha_nii_sii"),
        line_complexes=complexes,
        complex_statuses={
            key: str(value)
            for key, value in _from_key_values(
                object_row["complex_statuses"]
            ).items()
        },
        host_decomp_enabled=bool(object_row["host_decomp_enabled"]),
        total_spectrum=total_spectrum,
        host_model_on_quasar_grid=(
            np.asarray(row["host_model"], dtype=float)
            if row["host_model"] is not None else None
        ),
        warnings=warnings,
        metadata=workflow_metadata,
    )
    return workflow


def finalize_run(
    run: str | RunStore,
    *,
    compact_models: bool = False,
) -> Dict[str, str]:
    """Validate shards and materialize convenient compact Parquet files."""

    store = open_run(run) if isinstance(run, str) else run
    outputs = {}
    object_keys = store.read_table("objects", columns=["object_key"]).column(
        "object_key"
    ).to_pylist()
    duplicates = sorted(
        {key for key in object_keys if object_keys.count(key) > 1}
    )
    if duplicates:
        raise ValueError(f"Duplicate object keys in run: {duplicates[:10]}")
    for name in ("inputs", "objects", "measurements", "warnings", "failures", "derived"):
        table = store.read_table(name)
        output = store.path / "compact" / f"{name}.parquet"
        pq.write_table(table, output, compression="zstd")
        outputs[name] = str(output)
    if compact_models:
        output = store.path / "compact" / "models.parquet"
        pq.write_table(store.read_table("models"), output, compression="zstd")
        outputs["models"] = str(output)
    store.manifest["status"] = "complete"
    store.manifest["finalized_at"] = _now()
    store.manifest["compact_outputs"] = outputs
    store._write_manifest()
    return outputs


def build_science_catalog(
    run: str | RunStore,
    specification: Optional[Mapping[str, Mapping[str, Any]]] = None,
    *,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """Materialize a provisional wide catalog from long measurements."""

    store = open_run(run) if isinstance(run, str) else run
    objects = store.read_table("objects").to_pandas()
    if not specification:
        return objects
    measurements = store.read_table("measurements").to_pandas()
    catalog = objects.copy()
    for output_name, selector in specification.items():
        selected = measurements.copy()
        for key in ("section", "recipe_id", "feature_id", "role", "quantity"):
            if selector.get(key) is not None:
                selected = selected[selected[key] == selector[key]]
        values = selected.set_index("object_key")["value"]
        errors = selected.set_index("object_key")["error"]
        catalog[output_name] = catalog["object_key"].map(values)
        if selector.get("include_error", True):
            catalog[f"{output_name}_err"] = catalog["object_key"].map(errors)
    if output_path is not None:
        catalog.to_parquet(output_path, index=False)
    return catalog


def compute_derived_quantities(
    run: str | RunStore,
    calculators: Mapping[str, Callable[[Mapping[str, Any]], Any]],
) -> pd.DataFrame:
    """Run calibration-neutral user calculators and archive long-form results."""

    store = open_run(run) if isinstance(run, str) else run
    rows = []
    all_measurements = store.read_table("measurements").to_pylist()
    measurements_by_object: Dict[str, list[dict[str, Any]]] = {}
    for measurement in all_measurements:
        measurements_by_object.setdefault(
            measurement["object_key"], []
        ).append(measurement)
    for object_row in store.read_table("objects").to_pylist():
        key = object_row["object_key"]
        measurements = measurements_by_object.get(key, [])
        context = {"object": object_row, "measurements": measurements}
        for calibration_id, calculator in calculators.items():
            calculated = calculator(context)
            entries = calculated if isinstance(calculated, list) else [calculated]
            for entry in entries:
                rows.append(
                    {
                        "run_id": store.run_id,
                        "object_key": key,
                        "object_id": object_row["object_id"],
                        "quantity": str(entry["quantity"]),
                        "calibration_id": str(calibration_id),
                        "value": _float(entry.get("value")),
                        "statistical_error": _float(
                            entry.get("statistical_error")
                        ),
                        "intrinsic_scatter": _float(
                            entry.get("intrinsic_scatter")
                        ),
                        "total_error": _float(entry.get("total_error")),
                        "unit": entry.get("unit"),
                        "metadata": _key_values(entry.get("metadata", {})),
                    }
                )
    if rows:
        by_object: Dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_object.setdefault(row["object_key"], []).append(row)
        for object_rows in by_object.values():
            store.write_payload({"derived": object_rows})
    return pa.Table.from_pylist(rows, schema=SCHEMAS["derived"]).to_pandas()

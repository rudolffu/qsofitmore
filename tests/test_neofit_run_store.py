import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as pads
import pytest
from astropy.io import fits

from qsofitmore import neofit
from qsofitmore.host_decomp.io import SpectrumData


def _continuum_config():
    return neofit.GlobalContinuumConfig(
        uv_iron=None,
        optical_iron=None,
        balmer_continuum=neofit.BalmerContinuumConfig(enabled=False),
        balmer_series=neofit.BalmerSeriesConfig(enabled=False),
        clip_passes=0,
    )


def _spectrum_data(object_id="object-1", scale=1.0):
    wave = np.linspace(3500.0, 4500.0, 240)
    flux = scale * 2.0 * (wave / 4000.0) ** -1.1
    return SpectrumData(
        wave_obs=wave,
        flux=flux,
        error=np.full_like(wave, 0.05),
        redshift=0.0,
        object_id=object_id,
        ra=123.4,
        dec=-4.5,
        metadata={"input_file": f"memory-{object_id}"},
    )


def _parquet_input(path, count=2):
    rows = []
    for index in range(count):
        spectrum = _spectrum_data(f"object-{index}", 1.0 + 0.1 * index)
        rows.append(
            {
                "TARGETID": spectrum.object_id,
                "WAVELENGTH": spectrum.wave_obs.tolist(),
                "FLUX": spectrum.flux.tolist(),
                "ERROR": spectrum.error.tolist(),
                "Z": spectrum.redshift,
                "RA": spectrum.ra,
                "DEC": spectrum.dec,
            }
        )
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_single_object_bundle_round_trip_catalog_derived_and_qa(tmp_path):
    run = tmp_path / "single"
    result = neofit.fit_object_to_store(
        _spectrum_data(),
        str(run),
        global_config=_continuum_config(),
        complexes=[],
        write_qa=False,
    )

    assert Path(result.output_files["manifest"]).exists()
    assert Path(result.output_files["compact_models"]).exists()
    store = neofit.open_run(str(run))
    loaded = neofit.load_model(store, "object-1")
    np.testing.assert_allclose(loaded.spectrum.flux, result.spectrum.flux)
    np.testing.assert_allclose(loaded.continuum.model, result.continuum.model)
    assert store.read_table("objects").num_rows == 1
    assert store.read_table("models").num_rows == 1

    catalog = neofit.build_science_catalog(
        store,
        {
            "power_norm": {
                "section": "continuum_parameter",
                "quantity": "power_law.norm",
            }
        },
    )
    assert np.isfinite(catalog.loc[0, "power_norm"])

    before = store.read_table("models").num_rows
    derived = neofit.compute_derived_quantities(
        store,
        {
            "test-calibration": lambda context: {
                "quantity": "test_luminosity",
                "value": 42.0,
                "statistical_error": 0.1,
                "intrinsic_scatter": 0.2,
                "total_error": np.hypot(0.1, 0.2),
                "unit": "dex",
            }
        },
    )
    assert derived.loc[0, "value"] == pytest.approx(42.0)
    assert store.read_table("models").num_rows == before

    rendered = neofit.render_qa(
        store,
        object_ids=["object-1"],
        plot_config=neofit.GlobalQAPlotConfig(output_format="png"),
    )
    assert Path(rendered["object-1"]["global_plot"]).exists()


def test_serial_batch_resume_and_configuration_guard(tmp_path):
    source = tmp_path / "spectra.parquet"
    _parquet_input(source)
    run = tmp_path / "run"
    first = neofit.fit_batch(
        str(source),
        str(run),
        n_workers=1,
        global_config=_continuum_config(),
        complexes=[],
    )
    assert first.n_completed == 2
    assert first.n_failed == 0
    assert Path(first.compact_outputs["objects"]).exists()

    resumed = neofit.fit_batch(
        str(source),
        str(run),
        n_workers=1,
        global_config=_continuum_config(),
        complexes=[],
    )
    assert resumed.n_submitted == 0
    assert resumed.n_skipped == 2
    with pytest.raises(ValueError, match="immutable manifest"):
        neofit.fit_batch(
            str(source),
            str(run),
            n_workers=1,
            global_config=_continuum_config(),
            complexes=["mgii"],
        )


def test_parallel_batch_and_deterministic_multi_job_partition(tmp_path):
    source = tmp_path / "spectra.parquet"
    _parquet_input(source, count=4)
    parallel_run = tmp_path / "parallel"
    output = neofit.fit_batch(
        str(source),
        str(parallel_run),
        n_workers=2,
        task_size=1,
        global_config=_continuum_config(),
        complexes=[],
    )
    assert output.n_completed == 4
    assert neofit.open_run(str(parallel_run)).read_table("models").num_rows == 4

    sharded_run = tmp_path / "sharded"
    counts = []
    for shard_index in (0, 1):
        shard = neofit.fit_batch(
            str(source),
            str(sharded_run),
            n_workers=1,
            num_shards=2,
            shard_index=shard_index,
            finalize=False,
            global_config=_continuum_config(),
            complexes=[],
        )
        counts.append(shard.n_completed)
    assert sum(counts) == 4
    compact = neofit.finalize_run(str(sharded_run))
    assert len(pd.read_parquet(compact["objects"])) == 4


def test_failure_archive_keeps_input_locator(tmp_path):
    missing = neofit.SpectrumInput(
        source=str(tmp_path / "missing.fits"),
        object_id="missing",
        redshift=1.0,
    )
    run = tmp_path / "failed"
    output = neofit.fit_batch(
        [missing],
        str(run),
        n_workers=1,
        global_config=_continuum_config(),
        complexes=[],
    )
    assert output.n_failed == 1
    store = neofit.open_run(str(run))
    assert store.read_table("failures").num_rows == 1
    assert store.read_table("inputs").to_pylist()[0]["source"] == missing.source


def test_parquet_scanner_projects_case_insensitive_vector_columns(tmp_path):
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    _parquet_input(first, count=2)
    _parquet_input(second, count=1)
    records = list(
        neofit.scan_parquet_spectra(
            [str(first), str(second)],
            row_indices={str(first): [1], str(second): [0]},
            batch_size=1,
        )
    )
    assert [record[0].row_index for record in records] == [1, 0]
    assert [record[1].object_id for record in records] == [
        "object-1",
        "object-0",
    ]


def test_fits_reader_registry_handles_sdss_lamost_and_iraf(tmp_path):
    wave = np.linspace(4000.0, 4100.0, 20)
    flux = np.linspace(1.0, 2.0, 20)

    sdss = tmp_path / "sdss.fits"
    columns = [
        fits.Column(name="loglam", array=np.log10(wave), format="D"),
        fits.Column(name="flux", array=flux, format="D"),
        fits.Column(name="ivar", array=np.ones_like(flux), format="D"),
    ]
    fits.HDUList(
        [fits.PrimaryHDU(), fits.BinTableHDU.from_columns(columns)]
    ).writeto(sdss)
    assert neofit.detect_fits_reader(str(sdss)) == "sdss"
    np.testing.assert_allclose(
        neofit.read_spectrum(str(sdss), redshift=0.2).wave_obs, wave
    )

    lamost = tmp_path / "lamost.fits"
    columns = [
        fits.Column(name="WAVELENGTH", array=wave, format="D"),
        fits.Column(name="FLUX", array=flux, format="D"),
        fits.Column(name="ERROR", array=np.ones_like(flux), format="D"),
    ]
    fits.HDUList(
        [fits.PrimaryHDU(), fits.BinTableHDU.from_columns(columns)]
    ).writeto(lamost)
    assert neofit.detect_fits_reader(str(lamost)) == "lamost"
    np.testing.assert_allclose(
        neofit.read_spectrum(str(lamost), redshift=0.2).flux, flux
    )

    iraf = tmp_path / "iraf.fits"
    header = fits.Header()
    header["CRVAL1"] = wave[0]
    header["CDELT1"] = wave[1] - wave[0]
    fits.PrimaryHDU(flux, header=header).writeto(iraf)
    assert neofit.detect_fits_reader(str(iraf)) == "iraf"
    np.testing.assert_allclose(
        neofit.read_spectrum(str(iraf), redshift=0.2).wave_obs, wave
    )


def test_manifest_records_schema_and_shard_state(tmp_path):
    run = tmp_path / "manifest"
    neofit.fit_object_to_store(
        _spectrum_data(),
        str(run),
        global_config=_continuum_config(),
        complexes=[],
        write_qa=False,
    )
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["schema_version"]
    assert manifest["configuration_hash"]
    assert manifest["shard_state"]["models"] == 1
    assert pads.dataset(run / "models", format="parquet").count_rows() == 1

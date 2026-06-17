"""Import compatibility checks for the neofit MVP."""


def test_legacy_imports_still_work():
    import qsofitmore
    from qsofitmore import QSOFitNew

    assert qsofitmore is not None
    assert QSOFitNew is not None


def test_neofit_public_imports_work():
    from qsofitmore import neofit
    from qsofitmore.neofit import Spectrum, fit_line_complex

    assert neofit.Spectrum is Spectrum
    assert fit_line_complex is neofit.fit_line_complex

"""Regression coverage for psf_tools.analyze_channels.FLUOROPHORES.

Real dataset folders on /mmfs2/scratch/SDSMT.LOCAL/bscott/DataUpload/ use a
bare "Sca" token (no leading "m") for mScarlet, which the original
`msca|msc|scarlet|568|tdtomato|tdtom` pattern never matched -- silently
dropping the channel from get_extraction_plan(). See
20250905_py_FLM_2XFyve_Sca_memNG and 20250908_PY_FLM_2XFyve_Sca_memNG.
"""

from psf_tools.analyze_channels import parse_name_for_fluorophores

GFP = "GFP/mNG (488)"
MSCARLET = "mScarlet/CF568/TdTomato (561)"
CF647 = "CF647/SiR (640)"
FLM = "Fetal Liver Macrophage (flm)"
FYVE = "FYVE domain (PI3P probe, fused to GFP/mScarlet -- not a separate channel)"


def test_bare_sca_token_matches_mscarlet():
    # The confirmed-broken real folder names: bare "Sca", no "m" prefix.
    for name in (
        "20250905_py_FLM_2XFyve_Sca_memNG",
        "20250908_PY_FLM_2XFyve_Sca_memNG",
    ):
        found = parse_name_for_fluorophores(name)
        assert MSCARLET in found, f"{name!r} should detect mScarlet, got {found}"
        assert GFP in found


def test_sca_variant_matches_same_channels_as_full_spelling():
    # These sibling datasets spell mScarlet out in full (or as "mSca") and
    # were always detected correctly -- the "Sca"-only variant must resolve
    # to the same GFP/mNG + mScarlet pair, not a subset.
    reference = set(
        parse_name_for_fluorophores("20250915_py_FLM_2XFyve_mScarlet_mem_NG")
    )
    variant = set(parse_name_for_fluorophores("20250905_py_FLM_2XFyve_Sca_memNG"))
    assert {GFP, MSCARLET} <= reference
    assert {GFP, MSCARLET} <= variant


def test_bare_sca_pattern_does_not_false_positive_on_unrelated_words():
    # Words that merely *contain* "sc"/"sca" as a substring must not trigger
    # a false mScarlet match -- the fix uses a token-boundary guard, not an
    # unbounded substring search.
    for name in (
        "20251126-BLS_TetraspeckBeads",
        "20251124-BLS_FocalCheckBeads",
        "20251112_AH_deepred_PDMS",
    ):
        assert MSCARLET not in parse_name_for_fluorophores(name)


def test_hyphenated_and_concatenated_forms_still_match():
    # re.search on the whole lowercased string already handles these (no
    # hyphen requirement, contrary to the original bug report's hypothesis)
    # -- covered here as a regression guard.
    assert MSCARLET in parse_name_for_fluorophores(
        "20251019_AR_Phagocytosis_FLM_NG_SYK-mSc_HL60-CF647"
    )
    assert CF647 in parse_name_for_fluorophores(
        "20251019_AR_Phagocytosis_FLM_NG_SYK-mSc_HL60-CF647"
    )
    assert MSCARLET in parse_name_for_fluorophores("20250804-PY-FLM_memNG+2xFYVEmSC")


def test_fyve_is_reported_but_carries_no_wavelength():
    # FYVE is a probe domain fused to an existing FP, not its own excitation
    # line -- it must be visible in Detected_Channels for reporting, but its
    # standardized label must not contain any of the wavelength substrings
    # get_extraction_plan() checks for (488/405/561/640), or it would
    # silently inject a spurious output channel.
    found = parse_name_for_fluorophores("20250721-wtFLM_mSc2xFYVE")
    assert FYVE in found
    fyve_label = next(v for v in found if v == FYVE)
    for wavelength in ("488", "405", "561", "640"):
        assert wavelength not in fyve_label

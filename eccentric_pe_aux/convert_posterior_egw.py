"""
Convert posterior samples from an eccentric EOB run (eccentricity, mean_per_ano)
to GW-eccentricity (e_gw, mean_anomaly) measured at a reference frequency.
Supports both aligned-spin (e.g. SEOBNRv6EHM) and eccentric-precessing
(SEOBNRv6EPHM) approximants, and TEOBResumSDALI in either mode -- for TEOB the
spin treatment is taken from the posterior (or forced with --precessing).

TEOBResumSDALI has no backward integration, so f_ref must be chosen comfortably above
f_min. Starting the waveform lower is not an option: TEOB defines the spins at
f22_start, so that would change the physical system.
--extrapolate lifts that floor: e_gw is measured on a grid running upward from f_ref and
the value at f_ref is obtained by eccentric_pe_aux.egw_extrapolation.extrapolate_egw.

Can be run as a script: python -m eccentric_pe_aux.convert_posterior_egw --result <file>
"""

import argparse
import json
import os
import warnings
from copy import deepcopy
from multiprocessing import Pool

import astropy.units as u
import bilby
import lal
import numpy as np
import tqdm
from bilby.gw.conversion import bilby_to_lalsimulation_spins
from gw_eccentricity import measure_eccentricity
from eccentric_pe_aux.egw_extrapolation import extrapolate_egw
from pyseobnr.generate_waveform import GenerateWaveform
from lalsimulation.gwsignal import gwsignal_get_waveform_generator

warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

os.environ["OMP_NUM_THREADS"] = "1"

# Always driven with Cartesian spins, whatever the run's spin configuration.
PRECESSING_APPROXIMANTS = ["SEOBNRv6EPHM"]

# Usable either aligned or precessing; decided from the posterior unless --precessing.
DUAL_SPIN_APPROXIMANTS = ["TEOBResumSDALI"]


def posterior_is_precessing(posterior, tol: float = 1e-6) -> bool:
    """True if the posterior has a non-negligible in-plane spin component.
    Aligned runs store chi_1/chi_2 instead of tilts, or have tilts pinned to 0/pi.
    """
    if "tilt_1" not in posterior:
        return False
    s1 = (posterior["a_1"] * np.sin(posterior["tilt_1"])).abs().max()
    s2 = (posterior["a_2"] * np.sin(posterior["tilt_2"])).abs().max()
    return max(s1, s2) > tol


# Extra backward integration (units of M) added to the zero-ecc reference in the
# precessing case, where we cannot lower f22_start to give it a low-frequency
# margin (that would move f_ref and hence the spins).
PREC_QC_EXTRA_TBACK = 2000.0

# Grid used when --extrapolate is on: f_ref up to EXTRAP_FMAX_FACTOR * f_ref. Only the
# points gw_eccentricity can actually measure come back (in fref_out), so the floor does
# not have to be known in advance
EXTRAP_FMAX_FACTOR = 3.0
EXTRAP_NPOINTS = 16


def convert_to_egw(
    q: float,
    chi1,
    chi2,
    eccentricity: float,
    rel_anomaly: float,
    Mtot: float,
    f_min: float,
    deltaT: float,
    f_ref: float = 20.0,
    Mf_ref: float | None = None,
    t_back: float = 1000,
    method: str = "ResidualAmplitude",
    approximant: str = "SEOBNRv6EHM",
    precessing: bool = False,
    num_orbits_to_exclude_before_merger: int = 1,
    extra_kwargs: dict | None = None,
    debug: bool = False,
    extrapolate: bool = False,
    extrap_floor: float | None = None,
):
    """Generate an EOB waveform and measure the GW eccentricity at a reference frequency.

    Parameters
    ----------
    q : float
        Mass ratio m1/m2 >= 1.
    chi1, chi2 : float or array_like
        Spins of the two bodies. Aligned-spin (scalar, z-component) unless
        ``precessing`` is True, in which case each is the Cartesian spin
        vector ``[sx, sy, sz]`` defined at ``f_min``.
    eccentricity : float
        EOB eccentricity at f_min.
    rel_anomaly : float
        Relativistic anomaly at f_min.
    Mtot : float
        Total detector-frame mass (solar masses).
    f_min : float
        Starting GW frequency (Hz).
    deltaT : float
        Time step (s).
    f_ref : float, optional
        Reference GW frequency (Hz). Specify either f_ref or Mf_ref, not both.
    Mf_ref : float, optional
        Dimensionless reference frequency. Specify either f_ref or Mf_ref, not both.
    t_back : float, optional
        Duration of backwards integration (M).
    method : str, optional
        gw_eccentricity measurement method.
    approximant : str, optional
        Waveform approximant name.
    precessing : bool, optional
        If True, ``chi1``/``chi2`` are Cartesian spin vectors, all ell=2 modes
        are generated, and the eccentricity is measured from the coprecessing
        frame modes (``measure_eccentricity(..., precessing=True)``).
    num_orbits_to_exclude_before_merger : int, optional
        Number of orbits to exclude before merger in gw_eccentricity.
    extra_kwargs : dict or None, optional
        Extra keyword arguments passed to gw_eccentricity.measure_eccentricity.
    debug : bool, optional
        If True, show diagnostic plots from gw_eccentricity.
    extrapolate : bool, optional
        If True and f_ref lies below the lowest frequency gw_eccentricity can measure
        for this waveform, measure a grid above f_ref and extrapolate down to it (see
        the module docstring). If f_ref turns out to be directly measurable the grid
        value is used unchanged, so the flag is safe to leave on. ``mean_anomaly`` is
        returned as NaN whenever the value is extrapolated -- it is an angle, and the
        fit is meaningless for it.

    Returns
    -------
    e_gw : float
        GW eccentricity at f_ref.
    mean_anomaly : float
        Mean anomaly at f_ref.
    """
    m1 = q / (1.0 + q) * Mtot
    m2 = 1.0 / (1.0 + q) * Mtot

    if approximant == "TEOBResumSDALI":

        U = u.dimensionless_unscaled
        if precessing:
            # Requesting the (2,2)+(2,1) coprecessing modes makes TEOB twist them into
            # the full inertial ell=2 set, as gw_eccentricity needs when precessing.
            spin_params = {
                "spin1x": chi1[0] * U,
                "spin1y": chi1[1] * U,
                "spin1z": chi1[2] * U,
                "spin2x": chi2[0] * U,
                "spin2y": chi2[1] * U,
                "spin2z": chi2[2] * U,
            }
            mode_array = [(2, 2), (2, 1)]
        else:
            spin_params = {"spin1z": chi1 * U, "spin2z": chi2 * U}
            mode_array = [(2, 2)]

        parameters = {
            "mass1": m1 * u.solMass,
            "mass2": m2 * u.solMass,
            **spin_params,
            "f22_start": f_min * u.Hz,
            "eccentricity": eccentricity * U,
            "meanPerAno": rel_anomaly * u.rad,
            "ModeArray": mode_array,
            "deltaT": deltaT * u.s,
        }

        parameters_qc = deepcopy(parameters)
        parameters_qc["eccentricity"] = 0.0 * U
        if not precessing:
            # Margin for the zero-ecc reference; safe only for aligned spins, whose
            # components do not depend on the reference frequency.
            parameters_qc["f22_start"] = f_min * 0.9 * u.Hz
        # Precessing: f22_start is left alone -- TEOB defines the spins there, so
        # lowering it would change the binary. Get margin via a higher f_ref instead.

        gen = gwsignal_get_waveform_generator(approximant)
        modes = gen.generate_td_modes(**parameters)
        times = np.array(modes[2, 2].times)

        gen_qc = gwsignal_get_waveform_generator(approximant)
        modes_qc = gen_qc.generate_td_modes(**parameters_qc)
        times_qc = np.array(modes_qc[2, 2].times)

    else:

        if precessing:
            spin_params = {
                "spin1x": chi1[0],
                "spin1y": chi1[1],
                "spin1z": chi1[2],
                "spin2x": chi2[0],
                "spin2y": chi2[1],
                "spin2z": chi2[2],
            }
            # ell=2 coprecessing modes needed for the full inertial ell=2 set
            return_modes = [(2, 2), (2, 1)]
        else:
            spin_params = {"spin1z": chi1, "spin2z": chi2}
            return_modes = [(2, 2)]

        parameters = {
            "mass1": m1,
            "mass2": m2,
            **spin_params,
            "f22_start": f_min,
            "eccentricity": eccentricity,
            "rel_anomaly": rel_anomaly,
            "approximant": approximant,
            "return_modes": return_modes,
            "deltaT": deltaT,
            "t_backwards": t_back,
            "lmax_nyquist": 1,
            "warning_bwd_int": False,
        }

        parameters_qc = deepcopy(parameters)
        parameters_qc["eccentricity"] = 0.0
        if precessing:
            # We cannot lower f22_start here: eccentric approximants require
            # f_ref == f22_start, and the spins are defined at f_ref, so that
            # would make the zero-ecc reference a system with different spins.
            # We add a bit more backward integration instead so the
            # reference still spans the eccentric waveform.
            parameters_qc["t_backwards"] = t_back + PREC_QC_EXTRA_TBACK
        else:
            parameters_qc["f22_start"] = f_min * 0.9

        waveform = GenerateWaveform(parameters)
        times, modes = waveform.generate_td_modes()

        waveform_qc = GenerateWaveform(parameters_qc)
        times_qc, modes_qc = waveform_qc.generate_td_modes()

    # The coprecessing rotation needs the complete inertial ell=2 set, but TEOB falls
    # back to its aligned branch (only (2,+-2)) when the in-plane spin drops below ~1e-4,
    # as happens for samples with tilt -> 0. Follow what was actually generated: with no
    # in-plane spin there is no precession, so the (2,2) measurement is equivalent.
    ell2 = {(2, m) for m in (-2, -1, 0, 1, 2)}
    if precessing and not (
        ell2.issubset(set(modes.keys())) and ell2.issubset(set(modes_qc.keys()))
    ):
        precessing = False

    if precessing:
        # Pass the full inertial ell=2 set; gw_eccentricity rotates to the
        # coprecessing frame internally when precessing=True.
        hlm = {k: np.array(modes[k]) for k in modes if k[0] == 2}
        hlm_zeroecc = {k: np.array(modes_qc[k]) for k in modes_qc if k[0] == 2}
    else:
        hlm = {(2, 2): np.array(modes[2, 2])}
        hlm_zeroecc = {(2, 2): np.array(modes_qc[2, 2])}

    dataDict = {
        "t": times,
        "hlm": hlm,
        "t_zeroecc": times_qc,
        "hlm_zeroecc": hlm_zeroecc,
    }

    if f_ref is not None and Mf_ref is not None:
        raise ValueError("Specify only one of 'f_ref' or 'Mf_ref', not both.")

    if f_ref is None and Mf_ref is None:
        raise ValueError("You must specify at least one of 'f_ref' or 'Mf_ref'.")

    if f_ref is None and Mf_ref is not None:
        f_ref = Mf_ref / (Mtot * lal.MTSUN_SI)

    if extra_kwargs is None:
        extra_kwargs = {"treat_mid_points_between_pericenters_as_apocenters": True}

    fref_in = f_ref
    if extrapolate:
        fref_in = np.geomspace(f_ref, EXTRAP_FMAX_FACTOR * f_ref, EXTRAP_NPOINTS)

    return_dict = measure_eccentricity(
        fref_in=fref_in,
        method=method,
        dataDict=dataDict,
        precessing=precessing,
        num_orbits_to_exclude_before_merger=num_orbits_to_exclude_before_merger,
        extra_kwargs=extra_kwargs,
    )

    e_gw = return_dict["eccentricity"]
    mean_anomaly = return_dict["mean_anomaly"]

    if extrapolate:
        # gw_eccentricity returns only the grid points it could actually measure, so
        # fref_out -- not the requested grid -- is what the fit must be built on.
        f_out = np.atleast_1d(np.asarray(return_dict["fref_out"], float))
        e_out = np.atleast_1d(np.asarray(e_gw, float))
        a_out = np.atleast_1d(np.asarray(mean_anomaly, float))
        if len(f_out) != len(e_out):
            raise RuntimeError(
                f"fref_out ({len(f_out)}) and eccentricity ({len(e_out)}) disagree"
            )
        if extrap_floor is not None:
            keep = f_out >= extrap_floor
            if keep.sum() < 3:
                raise ValueError(
                    f"only {int(keep.sum())} measured points at or above the requested "
                    f"extrap_floor = {extrap_floor} Hz"
                )
            f_out, e_out, a_out = f_out[keep], e_out[keep], a_out[keep]

        hit = np.isclose(f_out, f_ref, rtol=1e-6)
        if hit.any():
            # directly measurable after all: no extrapolation, and the anomaly is real
            j = int(np.argmax(hit))
            e_gw, mean_anomaly = float(e_out[j]), float(a_out[j])
        else:
            e_gw = extrapolate_egw(f_out, e_out, f_ref)
            mean_anomaly = np.nan

    if debug:
        gwecc_object = return_dict["gwecc_object"]
        _, _ = gwecc_object.make_diagnostic_plots()

    return e_gw, mean_anomaly


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Compute e_gw from a bilby result file")
    p.add_argument(
        "--result", type=str, help="Bilby result file for an eccentric PE run"
    )
    p.add_argument("--n-cpu", type=int, help="Number of cores to use", default=64)
    p.add_argument(
        "--f-ref", type=float, help="Reference frequency in Hz", default=None
    )
    p.add_argument(
        "--Mf-ref", type=float, help="Dimensionless reference frequency", default=None
    )
    p.add_argument(
        "--t-back", type=float, help="Time for backwards integration", default=20000.0
    )
    p.add_argument("--srate", type=float, help="Sampling rate in Hz", default=16384.0)
    p.add_argument(
        "--method",
        type=str,
        help="gw_eccentricity method",
        default="ResidualAmplitude",  # ResidualAmplitude AmplitudeFits
    )
    p.add_argument(
        "--approximant",
        type=str,
        help="Approximant name",
        default="SEOBNRv6EHM",
    )
    p.add_argument(
        "--filename",
        type=str,
        help="Filename of the posterior",
        default="egw_converted_result.hdf5",
    )
    p.add_argument(
        "--num-orbits-to-exclude-before-merger",
        type=int,
        help="Number of orbits to exclude before merger in gw_eccentricity",
        default=1,
    )
    p.add_argument(
        "--extra-kwargs",
        type=str,
        help="JSON string of extra kwargs for gw_eccentricity.measure_eccentricity",
        default=None,
    )
    p.add_argument(
        "--n-samples",
        type=int,
        help="Number of randomly drawn samples to convert (default: all)",
        default=None,
    )
    p.add_argument(
        "--return-failures-as-nan",
        action="store_true",
        help="If set, samples that fail to convert will be set to NaN",
    )
    p.add_argument(
        "--extrapolate",
        action="store_true",
        help="Allow f_ref below the lowest measurable frequency: measure a grid above "
        "f_ref and extrapolate down to it (see module docstring). "
        "mean_anomaly_gw is NaN for extrapolated samples.",
    )
    p.add_argument(
        "--extrap-floor",
        type=float,
        default=None,
        help="With --extrapolate, ignore measured points below this frequency. Use it "
        "to reproduce another model's reach: two approximants on the same system have "
        "different lowest measurable frequencies, so validating one against the other "
        "must force both to extrapolate over the same span.",
    )
    p.add_argument(
        "--precessing",
        choices=["auto", "yes", "no"],
        default="auto",
        help="Spin treatment. 'auto' (default): always precessing for "
        f"{PRECESSING_APPROXIMANTS}, and for {DUAL_SPIN_APPROXIMANTS} decided from the "
        "posterior (precessing iff it has non-negligible in-plane spin).",
    )
    args = p.parse_args()

    result = bilby.read_in_result(args.result)
    pst = result.posterior
    meta = result.meta_data
    f_min = meta["likelihood"]["waveform_arguments"]["minimum_frequency"]

    if args.precessing == "auto":
        precessing = args.approximant in PRECESSING_APPROXIMANTS or (
            args.approximant in DUAL_SPIN_APPROXIMANTS and posterior_is_precessing(pst)
        )
    else:
        precessing = args.precessing == "yes"

    print(
        f"{args.approximant}: {'PRECESSING' if precessing else 'aligned-spin'} spin "
        f"treatment ({'forced by --precessing' if args.precessing != 'auto' else 'auto'})"
    )
    if precessing:
        # Spins are defined at the PE reference frequency
        f_ref_spins = meta["likelihood"]["waveform_arguments"]["reference_frequency"]
        print(
            f"Precessing approximant {args.approximant}: "
            f"reconstructing spins at f_ref = {f_ref_spins} Hz"
        )

    if args.n_samples is not None and args.n_samples < len(pst):
        idx = np.random.choice(len(pst), size=args.n_samples, replace=False)
        print(f"Randomly selected {args.n_samples} / {len(result.posterior)} samples")
        pst = pst.iloc[idx].reset_index(drop=True)
        result.posterior = pst

    print(f"f_min = {f_min} Hz")

    if args.f_ref is not None:
        print(f"Reference frequency f_ref = {args.f_ref} Hz")

    if args.Mf_ref is not None:
        print(f"Dimensionless reference frequency Mf_ref = {args.Mf_ref}")

    deltaT = 1 / args.srate

    if args.t_back > 0 and args.approximant in ["TEOBResumSDALI"]:
        print(
            "Warning: Backwards integration is not currently implemented for TEOBResumSDALI. Setting t_back to 0."
        )
        args.t_back = 0.0

    if args.approximant == "TEOBResumSDALI" and args.f_ref is not None:
        # The waveform starts at f_min and gw_eccentricity needs a few orbits before it
        # can locate the extrema, so the lowest measurable f_ref sits above f_min by a
        # margin that grows with eccentricity.
        if args.f_ref <= f_min and not args.extrapolate:
            raise SystemExit(
                f"f_ref = {args.f_ref} Hz is <= f_min = {f_min} Hz: every sample would "
                "fail. TEOBResumSDALI has no backward integration, so either pick a "
                "higher --f-ref (f_ref/f_min >~ 1.3 to start) or pass --extrapolate to "
                "measure above f_ref and extrapolate down. Starting the waveform lower "
                "is not an alternative: TEOB defines the spins at f22_start."
            )
        if args.f_ref < 1.2 * f_min and not args.extrapolate:
            print(
                f"Warning: f_ref = {args.f_ref} Hz is only {args.f_ref / f_min:.2f} x f_min; "
                "expect failures for some samples (raise --f-ref if so, or --extrapolate)."
            )
        if args.extrapolate:
            print(
                f"--extrapolate: measuring {EXTRAP_NPOINTS} points over "
                f"[{args.f_ref:.1f}, {EXTRAP_FMAX_FACTOR * args.f_ref:.1f}] Hz and "
                f"fitting down to {args.f_ref:.1f} Hz where it is not directly "
                "measurable (mean_anomaly_gw will be NaN for those samples)"
            )

    ALL_METHODS = [
        "ResidualAmplitude",
        "AmplitudeFits",
        "ResidualFrequency",
        "FrequencyFits",
        "Amplitude",
        "Frequency",
    ]

    def convert_to_egw_sample(i):
        if precessing:
            _, s1x, s1y, s1z, s2x, s2y, s2z = bilby_to_lalsimulation_spins(
                theta_jn=pst["theta_jn"][i],
                phi_jl=pst["phi_jl"][i],
                tilt_1=pst["tilt_1"][i],
                tilt_2=pst["tilt_2"][i],
                phi_12=pst["phi_12"][i],
                a_1=pst["a_1"][i],
                a_2=pst["a_2"][i],
                mass_1=pst["mass_1"][i] * lal.MSUN_SI,
                mass_2=pst["mass_2"][i] * lal.MSUN_SI,
                reference_frequency=f_ref_spins,
                phase=pst["phase"][i],
            )
            chi1, chi2 = [s1x, s1y, s1z], [s2x, s2y, s2z]
        elif "chi_1" in pst:
            chi1, chi2 = pst["chi_1"][i], pst["chi_2"][i]
        else:
            # aligned run stored in the (a, tilt) parameterisation: chi_z = a cos(tilt)
            chi1 = pst["a_1"][i] * np.cos(pst["tilt_1"][i])
            chi2 = pst["a_2"][i] * np.cos(pst["tilt_2"][i])

        methods_to_try = [args.method] + [m for m in ALL_METHODS if m != args.method]
        for method in methods_to_try:
            try:
                e_gw, mean_anomaly = convert_to_egw(
                    1 / pst["mass_ratio"][i],
                    chi1,
                    chi2,
                    pst["eccentricity"][i],
                    pst["mean_per_ano"][i],
                    pst["mass_1"][i] + pst["mass_2"][i],
                    f_min=f_min,
                    deltaT=deltaT,
                    f_ref=args.f_ref,
                    Mf_ref=args.Mf_ref,
                    t_back=args.t_back,
                    method=method,
                    approximant=args.approximant,
                    precessing=precessing,
                    num_orbits_to_exclude_before_merger=args.num_orbits_to_exclude_before_merger,
                    extra_kwargs=(
                        json.loads(args.extra_kwargs) if args.extra_kwargs else None
                    ),
                    extrapolate=args.extrapolate,
                    extrap_floor=args.extrap_floor,
                )
                return e_gw, mean_anomaly
            except Exception:
                continue
        return np.nan, np.nan

    e_gw_pst = []
    mean_anomaly_pst = []
    n_failed = 0

    with Pool(args.n_cpu) as pool:
        with tqdm.tqdm(total=len(pst)) as progress:
            for x, y in pool.imap(convert_to_egw_sample, range(len(pst))):
                if np.isnan(x):
                    n_failed += 1
                    if args.return_failures_as_nan:
                        e_gw_pst.append(x)
                        mean_anomaly_pst.append(y)
                else:
                    e_gw_pst.append(x)
                    mean_anomaly_pst.append(y)
                progress.update()

    print(f"\n{n_failed}/{len(pst)} samples failed ({100*n_failed/len(pst):.1f}%)")

    if n_failed / len(pst) > 0.5:
        raise RuntimeError(
            "More than 50% of samples failed to convert. Check the conversion settings."
        )

    e_gw_pst = np.array(e_gw_pst)
    mean_anomaly_pst = np.array(mean_anomaly_pst)

    result.posterior["e_gw"] = e_gw_pst
    result.posterior["mean_anomaly_gw"] = mean_anomaly_pst
    result.save_to_file(filename=args.filename)

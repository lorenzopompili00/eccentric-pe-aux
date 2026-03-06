"""
Convert posterior samples from SEOBNRv6EHM (eccentricity, mean_per_ano) to
GW-eccentricity (e_gw, mean_anomaly) measured at a reference frequency.

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
from gw_eccentricity import measure_eccentricity
from pyseobnr.generate_waveform import GenerateWaveform
from lalsimulation.gwsignal import gwsignal_get_waveform_generator

warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

os.environ["OMP_NUM_THREADS"] = "1"


def convert_to_egw(
    q: float,
    chi1: float,
    chi2: float,
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
    num_orbits_to_exclude_before_merger: int = 1,
    extra_kwargs: dict | None = None,
    debug: bool = False,
):
    """Generate an EOB waveform and measure the GW eccentricity at a reference frequency.

    Parameters
    ----------
    q : float
        Mass ratio m1/m2 >= 1.
    chi1, chi2 : float
        Aligned spins of the two bodies.
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
    num_orbits_to_exclude_before_merger : int, optional
        Number of orbits to exclude before merger in gw_eccentricity.
    extra_kwargs : dict or None, optional
        Extra keyword arguments passed to gw_eccentricity.measure_eccentricity.
    debug : bool, optional
        If True, show diagnostic plots from gw_eccentricity.

    Returns
    -------
    e_gw : float
        GW eccentricity at f_ref.
    mean_anomaly : float
        Mean anomaly at f_ref.
    """
    m1 = q / (1.0 + q) * Mtot
    m2 = 1.0 / (1.0 + q) * Mtot

    if approximant == 'TEOBResumSDALI':

        parameters = {
            "mass1": m1 * u.solMass,
            "mass2": m2 * u.solMass,
            "spin1z": chi1 * u.dimensionless_unscaled,
            "spin2z": chi2 * u.dimensionless_unscaled,
            "f22_start": f_min * u.Hz,
            "eccentricity": eccentricity * u.dimensionless_unscaled,
            "meanPerAno": rel_anomaly * u.rad,
            "ModeArray": [(2, 2)],
            "deltaT": deltaT * u.s,
        }

        parameters_qc = deepcopy(parameters)
        parameters_qc["eccentricity"] = 0.0 * u.dimensionless_unscaled
        parameters_qc["f22_start"] = f_min * 0.9 * u.Hz

        gen = gwsignal_get_waveform_generator(approximant)
        modes = gen.generate_td_modes(**parameters)
        times = np.array(modes[2, 2].times)

        gen_qc = gwsignal_get_waveform_generator(approximant)
        modes_qc = gen_qc.generate_td_modes(**parameters_qc)
        times_qc = np.array(modes_qc[2, 2].times)

    else:

        parameters = {
            "mass1": m1,
            "mass2": m2,
            "spin1z": chi1,
            "spin2z": chi2,
            "f22_start": f_min,
            "eccentricity": eccentricity,
            "rel_anomaly": rel_anomaly,
            "approximant": approximant,
            "return_modes": [(2, 2)],
            "deltaT": deltaT,
            "t_backwards": t_back,
            "lmax_nyquist": 1,
            "warning_bwd_int": False,
        }

        parameters_qc = deepcopy(parameters)
        parameters_qc["eccentricity"] = 0.0
        parameters_qc["f22_start"] = f_min * 0.9

        waveform = GenerateWaveform(parameters)
        times, modes = waveform.generate_td_modes()

        waveform_qc = GenerateWaveform(parameters_qc)
        times_qc, modes_qc = waveform_qc.generate_td_modes()

    dataDict = {
        "t": times,
        "hlm": {(2, 2): np.array(modes[2, 2])},
        "t_zeroecc": times_qc,
        "hlm_zeroecc": {(2, 2): np.array(modes_qc[2, 2])},
    }

    if f_ref is not None and Mf_ref is not None:
        raise ValueError("Specify only one of 'f_ref' or 'Mf_ref', not both.")

    if f_ref is None and Mf_ref is None:
        raise ValueError("You must specify at least one of 'f_ref' or 'Mf_ref'.")

    if f_ref is None and Mf_ref is not None:
        f_ref = Mf_ref / (Mtot * lal.MTSUN_SI)

    if extra_kwargs is None:
        extra_kwargs = {"treat_mid_points_between_pericenters_as_apocenters": True}

    return_dict = measure_eccentricity(
        fref_in=f_ref,
        method=method,
        dataDict=dataDict,
        num_orbits_to_exclude_before_merger=num_orbits_to_exclude_before_merger,
        extra_kwargs=extra_kwargs,
    )

    e_gw = return_dict["eccentricity"]
    mean_anomaly = return_dict["mean_anomaly"]

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
    p.add_argument(
        "--srate", type=float, help="Sampling rate in Hz", default=16384.0
    )
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
    args = p.parse_args()

    result = bilby.read_in_result(args.result)
    pst = result.posterior
    meta = result.meta_data
    f_min = meta["likelihood"]["waveform_arguments"]["minimum_frequency"]

    if args.n_samples is not None and args.n_samples < len(pst):
        idx = np.random.choice(len(pst), size=args.n_samples, replace=False)
        pst = pst.iloc[idx].reset_index(drop=True)
        print(f"Randomly selected {args.n_samples} / {len(result.posterior)} samples")

    print(f"f_min = {f_min} Hz")

    if args.f_ref is not None:
        print(f"Reference frequency f_ref = {args.f_ref} Hz")

    if args.Mf_ref is not None:
        print(f"Dimensionless reference frequency Mf_ref = {args.Mf_ref}")

    deltaT = 1 / args.srate

    if args.t_back > 0 and args.approximant in ["TEOBResumSDALI"]:
        print("Warning: Backwards integration is not currently implemented for TEOBResumSDALI. Setting t_back to 0.")
        args.t_back = 0.0

    def convert_to_egw_sample(i):
        try:
            (
                e_gw,
                mean_anomaly,
            ) = convert_to_egw(
                1 / pst["mass_ratio"][i],
                pst["chi_1"][i],
                pst["chi_2"][i],
                pst["eccentricity"][i],
                pst["mean_per_ano"][i],
                pst["mass_1"][i] + pst["mass_2"][i],
                f_min=f_min,
                deltaT=deltaT,
                f_ref=args.f_ref,
                Mf_ref=args.Mf_ref,
                t_back=args.t_back,
                method=args.method,
                approximant=args.approximant,
                num_orbits_to_exclude_before_merger=args.num_orbits_to_exclude_before_merger,
                extra_kwargs=json.loads(args.extra_kwargs) if args.extra_kwargs else None,
            )
            return e_gw, mean_anomaly
        except Exception:
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

import warnings

warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

import bilby
import numpy as np
import tqdm
import os
import argparse
import pandas as pd
import lal
from copy import deepcopy
from multiprocessing import Pool
from gw_eccentricity import measure_eccentricity
from pyseobnr.generate_waveform import GenerateWaveform
from bilby.gw.conversion import convert_to_lal_binary_black_hole_parameters

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
    Mf_ref: float = 0.01,
    t_back: float = 5000,
    method: str = "ResidualAmplitude",
    debug: bool = False,
):

    m1 = q / (1.0 + q) * Mtot
    m2 = 1.0 / (1.0 + q) * Mtot

    parameters = {
        "mass1": m1,
        "mass2": m2,
        "spin1z": chi1,
        "spin2z": chi2,
        "f22_start": f_min,
        "eccentricity": eccentricity,
        "rel_anomaly": rel_anomaly,
        "approximant": "SEOBNRv5EHM",
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
        "hlm": {(2, 2): modes[2, 2]},
        "t_zeroecc": times_qc,
        "hlm_zeroecc": {(2, 2): modes_qc[2, 2]},
    }

    if f_ref is not None and Mf_ref is not None:
        raise ValueError("Specify only one of 'f_ref' or 'Mf_ref', not both.")

    if f_ref is None and Mf_ref is None:
        raise ValueError("You must specify at least one of 'f_ref' or 'Mf_ref'.")

    if f_ref is None and Mf_ref is not None:
        f_ref = Mf_ref / (Mtot * lal.MTSUN_SI)

    return_dict = measure_eccentricity(
        fref_in=f_ref,
        method=method,
        dataDict=dataDict,
        num_orbits_to_exclude_before_merger=1,
        extra_kwargs={"treat_mid_points_between_pericenters_as_apocenters": True},
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
        "--t-back", type=float, help="Time for backwards integration", default=10000.0
    )
    p.add_argument(
        "--srate", type=float, help="Sampling rate in Hz", default=32768
    )
    p.add_argument(
        "--method",
        type=str,
        help="gw_eccentricity method",
        default="ResidualAmplitude",  # ResidualAmplitude AmplitudeFits
    )
    p.add_argument(
        "--filename",
        type=str,
        help="Filename of the posterior",
        default="egw_converted_result",
    )
    args = p.parse_args()

    result = bilby.read_in_result(args.result)
    pst = result.posterior
    meta = result.meta_data
    f_min = meta["likelihood"]["waveform_arguments"]["minimum_frequency"]
    deltaT = 1 / args.srate

    def convert_to_egw_sample(i):
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
        )

        return e_gw, mean_anomaly

    e_gw_pst = []
    mean_anomaly_pst = []

    with Pool(args.n_cpu) as p:
        with tqdm.tqdm(total=len(pst)) as progress:
            for x, y in p.imap(convert_to_egw_sample, range(len(pst))):
                e_gw_pst.append(x)
                mean_anomaly_pst.append(y)
                progress.update()

    e_gw_pst = np.array(e_gw_pst)
    mean_anomaly_pst = np.array(mean_anomaly_pst)

    result.posterior["e_gw"] = e_gw_pst
    result.posterior["mean_anomaly_gw"] = mean_anomaly_pst
    result.save_to_file(filename=args.filename)

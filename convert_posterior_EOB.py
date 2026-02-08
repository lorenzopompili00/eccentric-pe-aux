import warnings

warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

import bilby
import numpy as np
import tqdm
import os
import argparse
import pandas as pd
import lal
from multiprocessing import Pool
from scipy.interpolate import interp1d
from pyseobnr.generate_waveform import generate_modes_opt


os.environ["OMP_NUM_THREADS"] = "1"


def convert_to_eEOB(
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
    approximant: str = "SEOBNRv6EHM",
    debug: bool = False,
):

    omega_avg = f_min * Mtot * lal.MTSUN_SI * np.pi

    _, _, model = generate_modes_opt(
        q,
        chi1,
        chi2,
        omega_start = omega_avg,
        eccentricity = eccentricity,
        rel_anomaly = rel_anomaly,
        approximant = approximant,
        debug=True,
        settings=dict(return_modes=[(2,2)], dt=deltaT, lmax_nyquist=1)
    )
    t, r, phi, pr, pphi, e, z, x, H, Omega = model.dynamics.T

    interp_func_e = interp1d(x, e, kind='linear')
    interp_func_z = interp1d(x, z, kind='linear')

    if f_ref is not None and Mf_ref is not None:
        raise ValueError("Specify only one of 'f_ref' or 'Mf_ref', not both.")

    if f_ref is None and Mf_ref is None:
        raise ValueError("You must specify at least one of 'f_ref' or 'Mf_ref'.")

    if f_ref is None and Mf_ref is not None:
        f_ref = Mf_ref / (Mtot * lal.MTSUN_SI)


    omega_ref = np.pi * f_ref * Mtot * lal.MTSUN_SI
    x_ref = omega_ref**(2/3)
    e_ref = interp_func_e(x_ref)
    zeta_ref = interp_func_z(x_ref)

    if debug:
        import matplotlib.pyplot as plt

        plt.plot(x, e)
        plt.axvline(x=x_ref, color='gray', linestyle='--')
        plt.scatter([x_ref], [e_ref], color='red')
        plt.legend()
        plt.show()

        plt.plot(x, z)
        plt.axvline(x=x_ref, color='gray', linestyle='--')
        plt.scatter([x_ref], [zeta_ref], color='red')
        plt.legend()
        plt.show()

    return e_ref, np.mod(zeta_ref, 2 * np.pi)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Compute e_EOB from a bilby result file")
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
        "--srate", type=float, help="Sampling rate in Hz", default=32768
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
    args = p.parse_args()

    result = bilby.read_in_result(args.result)
    pst = result.posterior
    meta = result.meta_data
    f_min = meta["likelihood"]["waveform_arguments"]["minimum_frequency"]
    deltaT = 1 / args.srate

    def convert_to_eEOB_sample(i):
        (
            e_gw,
            mean_anomaly,
        ) = convert_to_eEOB(
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
            approximant=args.approximant,
        )

        return e_gw, mean_anomaly

    e_gw_pst = []
    mean_anomaly_pst = []

    with Pool(args.n_cpu) as p:
        with tqdm.tqdm(total=len(pst)) as progress:
            for x, y in p.imap(convert_to_eEOB_sample, range(len(pst))):
                e_gw_pst.append(x)
                mean_anomaly_pst.append(y)
                progress.update()

    e_gw_pst = np.array(e_gw_pst)
    mean_anomaly_pst = np.array(mean_anomaly_pst)

    result.posterior["e_EOB"] = e_gw_pst
    result.posterior["mean_anomaly_EOB"] = mean_anomaly_pst
    result.save_to_file(filename=args.filename)

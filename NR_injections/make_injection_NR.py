#!/usr/bin/env python
"""
Generate frame files for NR injections from SXS catalog.
"""

import argparse
import warnings
import yaml

warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

import numpy as np
import matplotlib.pyplot as plt
import sxs
import lal

from scipy.interpolate import interp1d
from gwpy.timeseries import TimeSeries
from pycbc.detector import Detector


def hp_hc_NR_phys_units(SXS_ID, injection_dict):
    """Compute h+ and hx in physical units from NR waveform."""
    iota, phi = injection_dict["iota"], injection_dict["phase"]
    mtot = injection_dict["total_mass"]
    dl = injection_dict["luminosity_distance"]

    wf = sxs.load(SXS_ID)
    reference_time = wf.metadata.reference_time
    w = wf.h
    reference_index = w.index_closest_to(reference_time)
    w = w[reference_index:, :]
    w = w.preprocess()

    hpc = 0.0
    for ell_m in w.LM:
        ell, m = ell_m
        hlm = w[:, w.index(ell, m)]
        ylm = lal.SpinWeightedSphericalHarmonic(iota, np.pi / 2 - phi, -2, ell, m)
        hpc += ylm * hlm

    hp = np.real(hpc)
    hc = -np.imag(hpc)

    hpc_times = w.t - w.max_norm_time()

    fac_times = mtot * lal.MTSUN_SI
    fac_h = (-1) * mtot * lal.MRSUN_SI / (dl * lal.PC_SI * 1e6)

    hpc_times *= fac_times
    hp *= fac_h
    hc *= fac_h

    return hp, hc, hpc_times


def compute_detector_times(ifo, inj_dict, hpc_times):
    """Compute time array for a specific detector."""
    deltaT = ifo.time_delay_from_earth_center(
        inj_dict["ra"], inj_dict["dec"], inj_dict["geocent_time"]
    )
    t_peak = inj_dict["geocent_time"] + deltaT
    times = hpc_times + t_peak
    return times


def compute_strain(ifo, inj_dict, times, hp, hc):
    """Compute detector strain from h+ and hx."""
    Fp, Fc = ifo.antenna_pattern(
        inj_dict["ra"], inj_dict["dec"], inj_dict["psi"], times
    )
    return Fp * hp + Fc * hc


def interpolate_to_grid(t_orig, h_orig, t_new):
    """Interpolate strain to a new time grid."""
    interp = interp1d(t_orig, h_orig, kind="cubic", bounds_error=False, fill_value=0.0)
    return interp(t_new)


def compute_interp_error_from_aligned(t_orig, h_orig, t_aligned, h_aligned, label):
    """Compute and plot interpolation error for debugging."""
    # Interpolate back from the aligned signal to original time points
    interp_back = interp1d(
        t_aligned, h_aligned, kind="cubic", bounds_error=False, fill_value=0.0
    )
    h_reconstructed = interp_back(t_orig)

    error = h_reconstructed - h_orig
    max_err = np.max(np.abs(error))
    rms_err = np.sqrt(np.mean(error**2))

    print(f"{label} interpolation error")
    print(f"Max Error: {max_err:.3e}")
    print(f"RMS Error: {rms_err:.3e}")

    plt.figure()
    plt.plot(t_orig, h_orig, label=f"{label} data")
    plt.plot(t_orig, error, label=f"{label} interpolation error")
    plt.xlabel("Time (s)")
    plt.ylabel("Error")
    plt.savefig(f"{label}_debug.png")
    plt.legend()
    plt.show()


def load_config(config_file):
    """Load configuration from YAML file."""
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Generate NR injection from the SXS catalog"
    )
    parser.add_argument("config", type=str, help="Path to configuration YAML file")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Extract configuration parameters
    SXS_ID = config["SXS_ID"]
    injection_dict = config["injection_dict"]
    debug = config.get("debug", False)
    output_prefix = config.get("output_prefix", SXS_ID.replace(":", "_"))
    channel_suffix = config.get("channel_suffix", "INJECTED")
    sampling_rate = config.get("sampling_rate", 2048.0)
    post_trigger_duration = config.get("post_trigger_duration", 4.0)
    duration = config.get("duration", 16.0)
    detector_names = config.get("detectors", ["H1", "L1"])

    # Create detector objects
    detectors = [Detector(name) for name in detector_names]

    print(f"Generating injection for {SXS_ID}")
    print(f"Total mass: {injection_dict['total_mass']} Msun")
    print(f"Luminosity distance: {injection_dict['luminosity_distance']} Mpc")
    print(f"Detectors: {', '.join(detector_names)}")

    # Generate h+ and hx in physical units
    hp, hc, hpc_times = hp_hc_NR_phys_units(SXS_ID, injection_dict)

    # Compute detector-specific times and strains
    detector_times = {}
    detector_strains = {}

    for det in detectors:
        times = compute_detector_times(det, injection_dict, hpc_times)
        strain = compute_strain(det, injection_dict, times, hp, hc)
        detector_times[det.name] = times
        detector_strains[det.name] = strain

    # RIFT requires that the time arrays extend between integer seconds with the
    # sampling rate at each IFO being a power of 2, so we will re-interpolate the
    # strain to a time arrays that satisfies these requirements.
    end_sec = np.ceil(injection_dict["geocent_time"]) + post_trigger_duration
    start_sec = end_sec - duration
    target_times = np.arange(start_sec, end_sec, 1 / sampling_rate)

    # Resample and write output
    for det in detectors:
        name = det.name
        times = detector_times[name]
        strain = detector_strains[name]

        resampled_strain = interpolate_to_grid(times, strain, target_times)

        if debug:
            compute_interp_error_from_aligned(
                times,
                strain,
                target_times,
                resampled_strain,
                f"{output_prefix}: {name}",
            )

        ts = TimeSeries(
            times=target_times,
            data=resampled_strain,
            channel=f"{name}:{channel_suffix}",
        )
        output_file = f"{output_prefix}_{name}.gwf"
        ts.write(output_file, format="gwf")
        print(f"Written: {output_file}")


if __name__ == "__main__":
    main()

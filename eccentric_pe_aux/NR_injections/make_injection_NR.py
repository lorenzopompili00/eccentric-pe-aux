#!/usr/bin/env python
"""
Generate frame files for NR injections from SXS catalog.
"""

import argparse
import warnings
import yaml

import lal
import matplotlib.pyplot as plt
import numpy as np
import os

import sxs
from gwpy.timeseries import TimeSeries
from pycbc.detector import Detector
from scipy.interpolate import interp1d

warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

# Restore the default plotting settings altered by gwpy
import matplotlib as mpl
_original_legend = plt.Axes.legend
def _patched_legend(self, *args, **kwargs):
    kwargs["handler_map"] = {mpl.lines.Line2D: mpl.legend_handler.HandlerLine2D()}
    return _original_legend(self, *args, **kwargs)
plt.Axes.legend = _patched_legend
mpl.rcParams.update(mpl.rcParamsDefault)


def hp_hc_NR_phys_units(SXS_ID, injection_dict, t_taper=500, sim_file=None):
    """Load an SXS waveform and return h+, hx in physical units.

    Sums all (ell, m) modes weighted by spin-weighted spherical harmonics,
    then rescales times and amplitudes from geometric to SI units.

    Parameters
    ----------
    SXS_ID : str
        SXS catalog identifier, e.g. ``"SXS:BBH:1359"``.
    injection_dict : dict
        Must contain ``iota``, ``phase``, ``total_mass`` (Msun),
        ``luminosity_distance`` (Mpc).
    t_taper : float, optional
        Duration (in M) over which to apply the start-of-waveform taper.

    Returns
    -------
    hp, hc : ndarray
        Plus and cross polarizations (strain).
    hpc_times : ndarray
        Time array (s), zero at peak amplitude.
    """
    iota, phi = injection_dict["iota"], injection_dict["phase"]
    mtot = injection_dict["total_mass"]
    dl = injection_dict["luminosity_distance"]

    if sim_file is not None:
        wf = sxs.load(sim_file)
        w = wf
    else:
        wf = sxs.load(SXS_ID)
        w = wf.h

    reference_time = wf.metadata.reference_time
    reference_index = w.index_closest_to(reference_time)
    w = w[reference_index:, :]
    w = w.preprocess(t1=reference_time, t2=reference_time + t_taper)

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
    plt.savefig(f"fig/{label}_debug.png")
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
    sim_id_label = SXS_ID.replace(":", "_").replace("/", "_")
    sim_file = config.get("sim_file", None)
    injection_dict = config["injection_dict"]
    debug = config.get("debug", False)
    plots = config.get("plots", False)
    output_prefix = config.get("output_prefix", sim_id_label)
    t_taper = config.get("t_taper", 500)
    channel_suffix = config.get("channel_suffix", "INJECTED")
    sampling_rate = config.get("sampling_rate", 2048.0)
    post_trigger_duration = config.get("post_trigger_duration", 4.0)
    duration = config.get("duration", 16.0)
    post_trigger_duration_bilby = config.get("post_trigger_duration_bilby", 2.0)
    duration_bilby = config.get("duration_bilby", 8.0)
    detector_names = config.get("detectors", ["H1", "L1"])

    if plots:
        os.makedirs("fig", exist_ok=True)

    # Define left and right boundaries of the data
    # [t_trigger + post_trigger - duration, t_trigger + post_trigger]
    left_b = injection_dict["geocent_time"] + post_trigger_duration - duration
    right_b = injection_dict["geocent_time"] + post_trigger_duration

    # Create detector objects
    detectors = [Detector(name) for name in detector_names]

    print(f"Generating injection for {SXS_ID}")
    print(f"Total mass: {injection_dict['total_mass']} Msun")
    print(f"Luminosity distance: {injection_dict['luminosity_distance']} Mpc")
    print(f"Detectors: {', '.join(detector_names)}")

    # Generate h+ and hx in physical units
    hp, hc, hpc_times = hp_hc_NR_phys_units(
        SXS_ID, injection_dict, t_taper=t_taper, sim_file=sim_file,
    )

    if plots:
        plt.plot(hpc_times, hp, label=r'$h_+$')
        plt.plot(hpc_times, hc, label=r'$h_\times$')
        plt.xlabel(r"$t$ [s]")
        plt.ylabel(r"$h$")
        plt.title(
            rf"{SXS_ID}, $M = {injection_dict['total_mass']}$, $d_L = {injection_dict['luminosity_distance']}$, $\iota = {injection_dict['iota']}$, $\phi = {injection_dict['phase']}$, $\mathrm{{dec}} = {injection_dict['dec']}$, $\mathrm{{ra}} = {injection_dict['ra']}$",
            y=1.06,    
        )
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(
            f"fig/polarizations_{sim_id_label}.png",
            bbox_inches="tight",
            dpi=400,
        )

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

        if plots:
            max_str = ts.times[np.argmax(np.asarray(ts.data))].to_value('s')
            fig, axs = plt.subplots(1, 2, figsize=(12, 4))
            axs[0].plot(ts.times, np.asarray(ts.data), label=ts.channel)
            axs[0].axvline(injection_dict["geocent_time"], color='gray', linewidth=1)
            axs[0].axvline(left_b, color='k', linewidth=1, linestyle='--')
            axs[0].axvline(right_b, color='k', linewidth=1, linestyle='--')
            axs[0].set_title(f"Interval: {right_b - left_b} [s]")
            axs[0].legend(frameon=False)
            axs[1].plot(ts.times, np.asarray(ts.data))
            axs[1].axvline(injection_dict["geocent_time"], color='gray', linewidth=1)
            axs[1].set_xlim(
                max_str + post_trigger_duration_bilby - duration_bilby,
                max_str + post_trigger_duration_bilby
            )
            axs[1].set_title(f"Interval: {duration_bilby} [s]")
            for ax in axs.flat:
                ax.grid(alpha=0.2)
            fig.suptitle(
                rf"{SXS_ID}, $M = {injection_dict['total_mass']}$, $d_L = {injection_dict['luminosity_distance']}$, $\iota = {injection_dict['iota']}$, $\phi = {injection_dict['phase']}$, $\mathrm{{dec}} = {injection_dict['dec']}$, $\mathrm{{ra}} = {injection_dict['ra']}$",
                y=1.01,
            )
            plt.savefig(
                f"fig/{name}_injection_{sim_id_label}.png",
                bbox_inches="tight",
                dpi=400,
            )


if __name__ == "__main__":
    main()

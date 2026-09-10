"""
Utilities for Canuda NR simulations.
Waveforms are stored as HDF5 files with one dataset per (l, m) mode.
"""

import re

import h5py
import lal
import numpy as np
from sxs.utilities import transition_function


def parse_parfile(parfile_text):
    """Extract a few physical parameters from the Einstein Toolkit parfile."""

    def get(key, cast=str):
        m = re.search(rf"^\s*{re.escape(key)}\s*=\s*([^\s#]+)", parfile_text, re.M)
        return cast(m.group(1).strip('"')) if m else None

    return {
        "mu": get("ScalarBase::mu", float),
        "ampSF": get("TwoPunctures_BBHSF::ampSF", float),
        "backreaction": get("TwoPunctures_BBHSF::switch_on_backreaction"),
        "m_plus": get("TwoPunctures_BBHSF::target_M_plus", float),
        "m_minus": get("TwoPunctures_BBHSF::target_M_minus", float),
    }


def load_modes(path):
    """Load all (l, m) modes, rescaled by M = m1 + m2 from the parfile.

    Returns
    -------
    t : ndarray
        Retarded time in units of M (common to all modes).
    modes : dict
        ``{(l, m): r*h_lm / M}`` (complex arrays).
    params : dict
        Output of :func:`parse_parfile`.
    """
    with h5py.File(path, "r") as f:
        params = parse_parfile(f["simulation/parfile"][()].decode())
        M = params["m_plus"] + params["m_minus"]
        modes = {}
        for name in f.attrs["modes"]:
            data = f[name][:]
            l, m = int(f[name].attrs["l"]), int(f[name].attrs["m"])
            modes[(l, m)] = (data[:, 1] + 1j * data[:, 2]) / M
        t = data[:, 0] / M  # same time grid for all modes
    return t, modes, params


def peak_time(t, modes):
    """Time of the maximum of sqrt(sum_lm |h_lm|^2)."""
    return t[np.argmax(sum(np.abs(h) ** 2 for h in modes.values()))]


def taper_modes(t, modes, t_start, t_taper):
    """Smoothly turn the modes on between ``t_start`` and ``t_start + t_taper``
    and off between 100 M and 200 M after the peak (as ``sxs`` ``preprocess``
    does by default), using the same C-infinity transition function as
    ``sxs.TimeSeries.taper``. Returns the time array and modes restricted to
    the non-zero window, and the peak time.
    """
    t_peak = peak_time(t, modes)
    t_end = t_peak + 200.0
    if t_start < t[0]:
        raise ValueError(f"t_start={t_start} is before the first data point {t[0]:.1f}")
    if t_start + t_taper >= t_peak:
        raise ValueError("taper end is after the peak, reduce t_start or t_taper")
    if t_end > t[-1]:
        raise ValueError(f"data ends at {t[-1]:.1f} M, less than 200 M after the peak")

    window = transition_function(t, t_start, t_start + t_taper) * transition_function(
        t, t_peak + 100.0, t_end, 1.0, 0.0
    )
    keep = (t >= t_start) & (t <= t_end)
    return t[keep], {lm: h[keep] * window[keep] for lm, h in modes.items()}, t_peak


def hp_hc_from_modes(t, modes, iota, phase):
    """``h_+ - i h_x = sum_lm Y_lm(iota, pi/2 - phase) h_lm``, with the same
    conventions as the SXS branch of ``make_injection_NR``."""
    hpc = np.zeros_like(t, dtype=complex)
    for (l, m), hlm in modes.items():
        hpc += lal.SpinWeightedSphericalHarmonic(iota, np.pi / 2 - phase, -2, l, m) * hlm
    return np.real(hpc), -np.imag(hpc)


def f22_at(t, modes, t_query, total_mass):
    """(2,2) GW frequency in Hz at time ``t_query`` (in M) for a total
    detector-frame mass ``total_mass`` (Msun). The likelihood's minimum
    frequency should be above its value at the end of the start taper."""
    phase = np.unwrap(np.angle(modes[(2, 2)]))
    Momega = np.abs(np.interp(t_query, t, np.gradient(phase, t)))
    return Momega / (2 * np.pi * total_mass * lal.MTSUN_SI)

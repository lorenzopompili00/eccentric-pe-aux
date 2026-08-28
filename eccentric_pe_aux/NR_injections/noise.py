"""
Noise handling for NR injections: Gaussian noise generation from a PSD and
real-data fetching, following bilby / bilby_pipe.
"""

import glob
import logging
import os

import lal
import numpy as np
from astropy.units import dimensionless_unscaled
from gwpy.timeseries import TimeSeries

logger = logging.getLogger(__name__)

# bilby_pipe defaults for the off-source PSD estimation segment, used to warn
# when the frame duration is too short for it in real noise
PSD_LENGTH = 32
PSD_MAXIMUM_DURATION = 1024


def get_power_spectral_density(det_name, psd_entry):
    """Build a bilby PowerSpectralDensity for one detector.

    Parameters
    ----------
    det_name : str
        Detector name, e.g. ``"H1"``.
    psd_entry : str or None
        PSD/ASD file path, or a bilby built-in noise-curve name (e.g.
        ``"aLIGO_O4_high_asd.txt"``). Files with ``"asd"`` in the name are
        read as amplitude spectral densities. ``None`` selects bilby's
        default sensitivity for the detector, as in bilby_pipe.

    Returns
    -------
    bilby.gw.detector.PowerSpectralDensity
    """
    from bilby.gw.detector import PowerSpectralDensity, get_empty_interferometer

    if psd_entry is None:
        return get_empty_interferometer(det_name).power_spectral_density
    if "asd" in os.path.basename(str(psd_entry)).lower():
        return PowerSpectralDensity.from_amplitude_spectral_density_file(psd_entry)
    return PowerSpectralDensity.from_power_spectral_density_file(psd_entry)


def gaussian_noise_time_domain(psd, sampling_rate, duration, start_time):
    """Draw a bilby-standard Gaussian noise realization in the time domain.

    Uses the same frequency-domain colored-noise generation as bilby /
    bilby_pipe. The noise is drawn from bilby's global random generator, so
    seed it with ``bilby.core.utils.random.seed`` for reproducibility.

    Parameters
    ----------
    psd : bilby.gw.detector.PowerSpectralDensity
    sampling_rate, duration, start_time : float

    Returns
    -------
    ndarray
        Noise strain of length ``int(duration * sampling_rate)``.
    """
    from bilby.gw.detector import InterferometerStrainData

    # The default InterferometerStrainData applies no frequency band mask
    strain_data = InterferometerStrainData()
    strain_data.set_from_power_spectral_density(
        psd,
        sampling_frequency=sampling_rate,
        duration=duration,
        start_time=start_time,
    )
    return np.asarray(strain_data.time_domain_strain)


def fetch_real_data(det_name, start_time, end_time, channel_dict, data_dict, sampling_rate):
    """Fetch real detector data following bilby_pipe's source priority.

    The channel ``"GWOSC"`` fetches public data; otherwise local frame files
    from ``data_dict`` are read, falling back to ``TimeSeries.get`` on the
    ``{det}:{channel}`` channel. Data is resampled to ``sampling_rate`` with
    lal (the bilby_pipe default resampling method).

    Parameters
    ----------
    det_name : str
    start_time, end_time : float
        GPS boundaries of the data to fetch.
    channel_dict : dict
        Input channel name per detector, or ``"GWOSC"``.
    data_dict : dict
        Optional local frame file path (or glob, or list of paths) per detector.
    sampling_rate : float

    Returns
    -------
    gwpy.timeseries.TimeSeries
    """
    channel = channel_dict.get(det_name)
    source = data_dict.get(det_name)

    if channel == "GWOSC":
        data = TimeSeries.fetch_open_data(
            det_name, start_time, end_time, sample_rate=16384
        )
    elif source is not None and channel is not None:
        if isinstance(source, str) and "*" in source:
            source = glob.glob(source)
        data = TimeSeries.read(source, f"{det_name}:{channel}")
        data = data.crop(start_time, end_time)
    elif channel is not None:
        data = TimeSeries.get(f"{det_name}:{channel}", start_time, end_time)
    else:
        raise ValueError(
            f"noise_type 'real' requires a channel_dict entry for {det_name}"
        )

    data = data.astype(np.float64)
    if data.unit != dimensionless_unscaled:
        logger.warning(
            "%s data has unit '%s', overwriting to dimensionless", det_name, data.unit
        )
        data.override_unit(dimensionless_unscaled)

    if data.sample_rate.value != sampling_rate:
        lal_timeseries = data.to_lal()
        lal.ResampleREAL8TimeSeries(lal_timeseries, float(1 / sampling_rate))
        data = TimeSeries(
            lal_timeseries.data.data,
            epoch=lal_timeseries.epoch,
            dt=lal_timeseries.deltaT,
        )
    return data


def optimal_snr(strain, sampling_rate, psd):
    """Optimal SNR of a time-domain strain given a PSD (bilby convention)."""
    hf = np.fft.rfft(strain) / sampling_rate
    freqs = np.fft.rfftfreq(len(strain), d=1 / sampling_rate)
    sn = psd.power_spectral_density_interpolated(freqs)
    mask = np.isfinite(sn) & (sn > 0) & (freqs > 0)
    df = freqs[1] - freqs[0]
    return np.sqrt(4 * df * np.sum(np.abs(hf[mask]) ** 2 / sn[mask]))

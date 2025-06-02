import lal
import numpy as np
from pyseobnr.eob.waveform.waveform_ecc import RadiationReactionForceEcc
from pyseobnr.generate_waveform import generate_modes_opt
from scipy.interpolate import interp1d


def eccentric_starting_frequency(f_start: float, e_low: float, M: float, q: float,
                                 chi_1: float, chi_2: float) -> float:
    """
    Compute the orbit-averaged starting frequency for gravitational waveform generation
    such that the instantaneous frequency at periastron matches the provided minimum frequency.

    Parameters:
        f_start (float): Minimum instantaneous frequency at periastron (Hz).
        e_low (float): Orbital eccentricity at f_low.
        M (float): Total detector-frame mass of the binary system (in solar masses).
        q (float): Mass ratio (m1/m2 > 1).
        chi_1 (float): Dimensionless spin of the first object.
        chi_2 (float): Dimensionless spin of the second object.

    Returns:
        f_low (float): Orbit-averaged starting frequency (Hz).
        e_start (float): Orbital eccentricity at f_start.
    """

    # Compute auxiliary variables
    m_1 = q / (1 + q)
    m_2 = 1 / (1 + q)
    nu = q / (1 + q) ** 2
    delta = m_1 - m_2
    chiS = m_1 * chi_1 + m_2 * chi_2
    chiA = m_1 * chi_1 - m_2 * chi_2

    # Convert frequency to dimensionless frequency
    omega_inst = f_start * M * lal.MTSUN_SI * np.pi

    # Initialize radiation reaction force object and evolution equations
    RR = RadiationReactionForceEcc()
    evolution = RR.evolution_equations

    evolution.initialize(
        chiA=chiA,
        chiS=chiS,
        delta=delta,
        flagPN1=1,
        flagPN2=1,
        flagPN3=1,
        flagPN32=1,
        flagPN52=1,
        nu=nu,
    )

    # Compute evolution at z=0 (periastron)
    evolution.compute(z=0, e=e_low, omega=omega_inst)

    # Get orbit-averaged omega and convert back to frequency
    omega_avg = evolution.get("xavg_omegainst") ** (3 / 2)
    f_low = omega_avg / (M * lal.MTSUN_SI * np.pi)

    _, _, model = generate_modes_opt(
        q,
        chi_1,
        chi_2,
        omega_start = omega_avg,
        eccentricity = e_low,
        rel_anomaly = 0.0,
        approximant = "SEOBNRv5EHM",
        debug=True,
        settings=dict(return_modes=[(2,2)], lmax_nyquist=1)
    )
    t, r, phi, pr, pphi, e, z, x, H, Omega = model.dynamics.T

    interp_func_e = interp1d(x, e, kind='linear')

    omega_start = np.pi * f_start * M * lal.MTSUN_SI
    x_start = omega_start**(2/3)
    e_start = interp_func_e(x_start)

    return f_low, e_start

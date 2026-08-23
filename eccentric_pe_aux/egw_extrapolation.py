"""Extrapolating a measured e_gw(f) curve below the lowest frequency it can be measured at.

gw_eccentricity needs a few radial cycles below the reference frequency to locate the
extrema, so e_gw is never measurable at the very start of a waveform. Two situations hit
this: an approximant that cannot be started below the frequency where its parameters are
anchored, and NR, whose e_gw(f) curve only becomes valid a few orbits after the
relaxation time.

The extrapolation divides the measured curve by the Peters (1964) relation and fits what
is left. Along a Newtonian inspiral f = K g(e) with K constant for any eccentricity, so
once g is divided out the residual K drifts only slowly and a straight line in ln K
against ln f carries it below the measured range.
"""

import numpy as np
from scipy.optimize import brentq


def peters_g(e):
    """f / K along a Peters (1964) inspiral, exact at Newtonian order for any e.

    Monotonically decreasing in e, so ``g(e) = target`` has at most one root.
    """
    e = np.asarray(e, float)
    return (
        e ** (12 / 19) * (1 + 121 * e**2 / 304) ** (870 / 2299) / (1 - e**2)
    ) ** -1.5


def extrapolate_egw(f, e, f_target, band=2.0):
    """e_gw at ``f_target`` from a measured ``(f, e)`` curve lying entirely above it.

    Parameters
    ----------
    f, e : array_like
        The measured curve. Only the points within ``band`` of ``min(f)`` are fitted --
        the far end of the curve carries no information about the extrapolated end and
        only drags the fit.
    f_target : float
        Frequency to extrapolate to; expected below ``min(f)``.
    band : float
        Width of the fitted window, as a multiple of ``min(f)``. Widen it when the curve
        is sparse and a narrow window leaves too few points; narrow it when the curve is
        densely sampled and spans a wide range in e, where a linear ln K drift stops
        describing the whole window.

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If fewer than three points fall inside the window, or if the fitted K puts the
        target outside the range of g so the inversion has no root.
    """
    f, e = np.asarray(f, float), np.asarray(e, float)
    m = f <= band * f.min()
    if m.sum() < 3:
        raise ValueError(
            f"need >= 3 measured points within {band}x of {f.min():.2f} Hz to "
            f"extrapolate, got {int(m.sum())}"
        )
    c = np.polyfit(np.log(f[m]), np.log(f[m] / peters_g(e[m])), 1)
    target = f_target / np.exp(np.polyval(c, np.log(f_target)))
    try:
        return float(brentq(lambda x: peters_g(x) - target, 1e-6, 1 - 1e-6))
    except (ValueError, RuntimeError) as exc:
        raise ValueError(
            f"Peters inversion has no root for f_target = {f_target} Hz "
            f"(fitted g = {target:.4g}); the curve is probably too short or too noisy "
            "to extrapolate this far"
        ) from exc

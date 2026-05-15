"""
Astrophysically motivated single-single GW-capture prior for bilby PE,
based on Gupte et al. 2026 (arXiv:2603.29019).  

The prior is conditional on the symmetric mass ratio, expressed as a bilby
``ConditionalPriorDict`` chain (``nu`` from the sampled ``mass_ratio``,
``sigma`` a sampled host-dispersion parameter the user supplies a prior for).
A more detailed description is in the note ``capture_prior.md``.

Provides
--------
CaptureEnergyPrior : ConditionalGamma subclass   -> ``delta_energy`` = E_0 - 1
CaptureMomentumPrior : ConditionalPowerLaw subclass -> ``momentum`` = p_phi^0
make_capture_prior_dict : build a ready ``ConditionalPriorDict``
convert_to_capture_bbh_parameters : waveform-generator conversion (+ ``energy``)
generate_all_capture_bbh_parameters : post-processing generation function
CaptureHyperbolicWaveformGenerator : generator with the conversion baked in

"""

import lal
import numpy as np

from bilby.core.prior import (
    ConditionalGamma,
    ConditionalPowerLaw,
    ConditionalPriorDict,
    Constraint,
    Uniform,
)
from bilby.gw.prior import (
    UniformInComponentsChirpMass,
    UniformInComponentsMassRatio,
)
from bilby.gw.conversion import (
    convert_to_lal_binary_black_hole_parameters,
    generate_all_bbh_parameters,
)

from .hyperbolic import HyperbolicGWSignalWaveformGenerator

# --- constants --------------------------------------------------------------
C_KMS = lal.C_SI / 1.0e3                          # speed of light, km/s
PETERS_K = 85.0 * np.pi / (6.0 * np.sqrt(2.0))    # Peters capture coefficient
                                                  # (dimensionless formula, ~31.47)


def nu_of_q(mass_ratio):
    """Symmetric mass ratio nu = q/(1+q)^2 (symmetric under q <-> 1/q)."""
    q = np.asarray(mass_ratio, dtype=float)
    return q / (1.0 + q) ** 2


# ---------------------------------------------------------------------------
# Module-level condition functions.
# The arguments after `reference_params` declare the required variables, which
# bilby uses to resolve the sampling order automatically.
# ---------------------------------------------------------------------------


def _condition_capture_energy(reference_params, mass_ratio, sigma):
    """p(E_0 - 1 | nu, sigma) = Gamma(k = 5/7, theta = 2 nu (sigma/c)^2).

    `sigma` is the host velocity dispersion in km/s (a sampled parameter).
    """
    nu = nu_of_q(mass_ratio)
    theta = 2.0 * nu * (np.asarray(sigma, dtype=float) / C_KMS) ** 2
    return dict(k=reference_params["k"], theta=theta)


def _condition_capture_momentum(reference_params, delta_energy, mass_ratio):
    """p(p_phi^0 | E_0-1, nu) = PowerLaw(alpha=1) on [minimum, sqrt(2 r_p,max_hat)].

    r_p,max_hat = [ (85 pi / 6 sqrt2) nu / beta^2 ]^(2/7), beta^2 = 2 (E_0-1)/nu.
    """
    nu = nu_of_q(mass_ratio)
    beta2 = 2.0 * np.asarray(delta_energy, dtype=float) / nu
    rp_max_hat = (PETERS_K * nu / beta2) ** (2.0 / 7.0)
    p_phi_max = np.sqrt(2.0 * rp_max_hat)
    return dict(
        alpha=reference_params["alpha"],
        minimum=reference_params["minimum"],
        maximum=p_phi_max,
    )


# ---------------------------------------------------------------------------
# Conditional prior subclasses
# ---------------------------------------------------------------------------


class CaptureEnergyPrior(ConditionalGamma):
    r"""Conditional prior for ``delta_energy`` = :math:`E_0 - 1`.

    :math:`p(E_0-1\mid\nu,\sigma)=\mathrm{Gamma}(k=5/7,\ \theta=2\nu(\sigma/c)^2)`,
    with :math:`\nu` from ``mass_ratio`` and :math:`\sigma` from ``sigma``
    (both sampled).  Use :func:`convert_to_capture_bbh_parameters` to obtain
    the waveform ``energy`` = 1 + ``delta_energy``.

    Parameters
    ----------
    k : float, optional
        Gamma shape; the capture-biased Maxwellian gives ``5/7``.
    """

    def __init__(self, k=5.0 / 7.0, name="delta_energy",
                 latex_label=r"$E_0-1$", unit=None, boundary=None):
        super().__init__(
            condition_func=_condition_capture_energy,
            name=name, latex_label=latex_label, unit=unit, boundary=boundary,
            k=float(k), theta=1.0,
        )
        # bilby's conditional_prior_factory rewrites __name__/__qualname__ to
        # "ConditionalGamma" at instantiation; restore them so pickle (used by
        # bilby_pipe's data dump) resolves the class back to this subclass.
        type(self).__name__ = "CaptureEnergyPrior"
        type(self).__qualname__ = "CaptureEnergyPrior"

    def __repr__(self):
        # Clean, prior-file-round-trippable repr (condition_func is implicit).
        cls = f"{self.__class__.__module__}.{self.__class__.__name__}"
        return (f"{cls}(k={self.k}, name={self.name!r}, "
                f"latex_label={self.latex_label!r}, unit={self.unit!r}, "
                f"boundary={self.boundary!r})")


class CaptureMomentumPrior(ConditionalPowerLaw):
    r"""Conditional prior for ``momentum`` = :math:`p_\phi^0`.

    :math:`p(p_\phi^0\mid E_0,\nu)\propto p_\phi^0` on
    :math:`[\,\texttt{minimum},\ \sqrt{2\,\hat r_{p,\max}}\,]` — uniform
    pericenter in the gravitational-focusing limit mapped through
    :math:`p_\phi^0=\sqrt{2\hat r_p}`.  Depends on ``delta_energy`` and
    ``mass_ratio``.

    Parameters
    ----------
    minimum : float, optional
        Lower bound on :math:`p_\phi^0` (default 0; a tiny floor is harmless).
    """

    def __init__(self, minimum=0.0, name="momentum",
                 latex_label=r"$p_\phi^0$", unit=None, boundary=None):
        super().__init__(
            condition_func=_condition_capture_momentum,
            name=name, latex_label=latex_label, unit=unit, boundary=boundary,
            alpha=1.0, minimum=float(minimum), maximum=1.0,
        )
        # See CaptureEnergyPrior: restore the name mangled by the factory so
        # the instance is picklable by reference (bilby_pipe data dump).
        type(self).__name__ = "CaptureMomentumPrior"
        type(self).__qualname__ = "CaptureMomentumPrior"

    def __repr__(self):
        # Clean, prior-file-round-trippable repr (alpha=1 and the dynamic
        # maximum are implicit; condition_func is implicit).
        cls = f"{self.__class__.__module__}.{self.__class__.__name__}"
        return (f"{cls}(minimum={self.minimum}, name={self.name!r}, "
                f"latex_label={self.latex_label!r}, unit={self.unit!r}, "
                f"boundary={self.boundary!r})")


# ---------------------------------------------------------------------------
# Convenience: ready-made ConditionalPriorDict
# ---------------------------------------------------------------------------


def make_capture_prior_dict(
    sigma_prior=None,
    chirp_mass_minimum=40.0,
    chirp_mass_maximum=200.0,
    mass_ratio_minimum=0.125,
    mass_ratio_maximum=1.0,
    momentum_minimum=0.0,
):
    """Build a :class:`ConditionalPriorDict` with the capture-IC priors.

    Includes only the intrinsic parameters needed to exercise the capture
    prior (``sigma``, ``chirp_mass``, ``mass_ratio``, ``delta_energy``,
    ``momentum``); add extrinsic priors as usual for a full run.

    Parameters
    ----------
    sigma_prior : bilby Prior, optional
        Prior on the host velocity dispersion (km/s).  Default
        ``Uniform(0.1, 1000)`` km/s (a broad, conservative hyper-prior).
        Pass any bilby prior (e.g. ``LogUniform``) to change it.
    """
    if sigma_prior is None:
        sigma_prior = Uniform(name="sigma", minimum=0.1, maximum=1000.0,
                              latex_label=r"$\sigma\,[\mathrm{km/s}]$")
    priors = ConditionalPriorDict()
    priors["sigma"] = sigma_prior
    priors["chirp_mass"] = UniformInComponentsChirpMass(
        minimum=chirp_mass_minimum, maximum=chirp_mass_maximum,
        name="chirp_mass", unit=r"$M_{\odot}$",
    )
    priors["mass_ratio"] = UniformInComponentsMassRatio(
        minimum=mass_ratio_minimum, maximum=mass_ratio_maximum,
        name="mass_ratio",
    )
    priors["mass_1"] = Constraint(name="mass_1", minimum=1, maximum=1000)
    priors["mass_2"] = Constraint(name="mass_2", minimum=1, maximum=1000)
    priors["delta_energy"] = CaptureEnergyPrior()
    priors["momentum"] = CaptureMomentumPrior(minimum=momentum_minimum)
    return priors


# ---------------------------------------------------------------------------
# Waveform-generator conversion / post-processing
# ---------------------------------------------------------------------------


def _add_energy_from_delta(sample, added_keys=None):
    """energy = 1 + delta_energy (ConditionalGamma lives on [0, inf))."""
    if added_keys is None:
        added_keys = []
    if "delta_energy" in sample and "energy" not in sample:
        sample["energy"] = 1.0 + sample["delta_energy"]
        added_keys = added_keys + ["energy"]
    return sample, added_keys


def convert_to_capture_bbh_parameters(parameters):
    """Waveform-generator conversion: standard LAL BBH + ``energy`` injection.

    Returns ``(converted_parameters, added_keys)``.
    """
    converted_parameters, added_keys = convert_to_lal_binary_black_hole_parameters(
        parameters
    )
    converted_parameters, added_keys = _add_energy_from_delta(
        converted_parameters, added_keys
    )
    return converted_parameters, added_keys


def generate_all_capture_bbh_parameters(
    sample, likelihood=None, priors=None, npool=1
):
    """Fill in all missing BBH parameters and add ``energy`` = 1 + delta_energy."""
    output_sample = generate_all_bbh_parameters(sample, likelihood, priors, npool)
    output_sample, _ = _add_energy_from_delta(output_sample)
    return output_sample


class CaptureHyperbolicWaveformGenerator(HyperbolicGWSignalWaveformGenerator):
    """Hyperbolic generator with the capture conversion baked in.

    Adds ``energy`` = 1 + ``delta_energy`` before calling the waveform model;
    for use with parallel_bilby (which mishandles custom conversion funcs).
    """

    def __init__(self, **kwargs):
        kwargs["parameter_conversion"] = convert_to_capture_bbh_parameters
        super().__init__(**kwargs)

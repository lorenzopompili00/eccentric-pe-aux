"""
Cartesian eccentricity vector parameterization for bilby PE.

Replaces the (eccentricity, mean_per_ano) polar parameterization with
Cartesian variables (ecc_x, ecc_y) = (e cos l, e sin l),
removing the coordinate singularity at e=0 where the anomaly is degenerate.

Provides
--------
EccentricityVectorDist : BaseJointPriorDist
    Joint prior distribution on (ecc_x, ecc_y) in a disk.
EccentricityVectorPrior : JointPrior
    Per-parameter wrapper for use in prior files.
CartesianEccWaveformGenerator : GWSignalWaveformGenerator
    Waveform generator that converts (ecc_x, ecc_y) -> (eccentricity, mean_per_ano).
bbh_ecc_cartesian_conversion : callable
    Prior-dict conversion function that extends the standard BBH conversion
    with the (ecc_x, ecc_y) -> eccentricity derivation.  Use as
    `conversion_function` in the prior file so that the `eccentricity`
    Constraint is evaluated.

Usage in prior file
-------------------
    ecc_vector = eccentric_pe_aux.bilby_aux.ecc_cartesian.EccentricityVectorDist(names=['ecc_x', 'ecc_y'], ecc_max=0.4)
    ecc_x = eccentric_pe_aux.bilby_aux.ecc_cartesian.EccentricityVectorPrior(dist=ecc_vector, name='ecc_x', latex_label='$e_x$')
    ecc_y = eccentric_pe_aux.bilby_aux.ecc_cartesian.EccentricityVectorPrior(dist=ecc_vector, name='ecc_y', latex_label='$e_y$')

Usage in config.ini
-------------------
    conversion_function = eccentric_pe_aux.bilby_aux.ecc_cartesian.bbh_ecc_cartesian_conversion
    waveform-generator = eccentric_pe_aux.bilby_aux.ecc_cartesian.CartesianEccWaveformGenerator
"""

import numpy as np

from bilby.core.prior.joint import BaseJointPriorDist, JointPrior, JointPriorDistError
from bilby.core.utils import random
from bilby.gw.waveform_generator import GWSignalWaveformGenerator
from bilby.gw.conversion import (
    convert_to_lal_binary_black_hole_parameters,
    generate_mass_parameters,
)


# ---------------------------------------------------------------------------
# Prior-dict conversion function (for Constraint evaluation)
# ---------------------------------------------------------------------------

def _add_eccentricity_from_cartesian(sample):
    """Add `eccentricity` and `mean_per_ano` to `sample` in-place."""
    if "ecc_x" in sample and "ecc_y" in sample:
        sample["eccentricity"] = np.sqrt(
            sample["ecc_x"] ** 2 + sample["ecc_y"] ** 2
        )
        sample["mean_per_ano"] = np.arctan2(
            sample["ecc_y"], sample["ecc_x"]
        ) % (2 * np.pi)
    return sample


def bbh_ecc_cartesian_conversion(sample):
    """BBH conversion function extended with (ecc_x, ecc_y) -> eccentricity.

    Drop-in replacement for `BBHPriorDict.default_conversion_function`.
    Suitable for use as `conversion_function` in a `.prior` file so that
    a `Constraint` on `eccentricity` is properly evaluated.
    """
    sample = _add_eccentricity_from_cartesian(sample)
    sample, _ = convert_to_lal_binary_black_hole_parameters(sample)
    sample = generate_mass_parameters(sample)
    return sample


# ---------------------------------------------------------------------------
# Joint prior distribution
# ---------------------------------------------------------------------------

class EccentricityVectorDist(BaseJointPriorDist):
    r"""Joint prior on the Cartesian eccentricity vector (ecc_x, ecc_y).

    Two modes are available, controlled by `flat_in_eccentricity`:

    * `flat_in_eccentricity=True` (default) — the rescale maps to a distribution
      with :math:`p(e) = 1/e_{\max}` (uniform in eccentricity).  The 2-D
      density is :math:`p(e_x, e_y) \propto 1/\sqrt{e_x^2 + e_y^2}`, which
      diverges at the origin but is integrable.
    
    * `flat_in_eccentricity=False` — uniform on the disk.
      The induced 1-D marginal prior on eccentricity is :math:`p(e) \propto e`.

    Parameters
    ----------
    names : list of str
        Names of the two Cartesian components, e.g. `['ecc_x', 'ecc_y']`.
    ecc_max : float
        Maximum eccentricity (disk radius).
    flat_in_eccentricity : bool, optional
        If True, enforce uniform prior on eccentricity. Default True.
    """

    def __init__(self, names, ecc_max, flat_in_eccentricity=True):
        self.ecc_max = float(ecc_max)
        self.flat_in_eccentricity = flat_in_eccentricity

        if len(names) != 2:
            raise ValueError("EccentricityVectorDist requires exactly 2 names")

        bounds = [(-self.ecc_max, self.ecc_max)] * 2
        super().__init__(names=names, bounds=bounds)
        self.distname = "ecc_vector"

        # Pre-compute log-normalization
        if self.flat_in_eccentricity:
            # p(ecc_x, ecc_y) = 1/(2*pi*ecc_max*e) inside disk
            self._log_norm = -np.log(2 * np.pi * self.ecc_max)
        else:
            # p(ecc_x, ecc_y) = 1/(pi*ecc_max^2) inside disk
            self._log_norm = -np.log(np.pi * self.ecc_max ** 2)

    # -- Rescale: unit hypercube -> disk ------------------------------------

    def _rescale(self, samp, **kwargs):
        """Map [0,1]^2 samples to the disk.

        Parameters
        ----------
        samp : ndarray, shape (N, 2)
            Columns are unit-cube draws for the two Cartesian components.

        Returns
        -------
        ndarray, shape (N, 2)
        """
        u1 = samp[:, 0]
        u2 = samp[:, 1]

        if self.flat_in_eccentricity:
            r = self.ecc_max * u1           # p(r) = 1/ecc_max  => p(e) uniform
        else:
            r = self.ecc_max * np.sqrt(u1)  # p(r) ~ r          => uniform on disk

        theta = 2 * np.pi * u2

        out = np.empty_like(samp)
        out[:, 0] = r * np.cos(theta)
        out[:, 1] = r * np.sin(theta)
        return out

    # -- Direct sampling ----------------------------------------------------

    def _sample(self, size, **kwargs):
        u = random.rng.uniform(0, 1, size=(size, 2))
        return self._rescale(u)

    # -- Log-probability ----------------------------------------------------

    def _ln_prob(self, samp, lnprob, outbounds):
        """Evaluate log p(ecc_x, ecc_y).

        Parameters
        ----------
        samp : ndarray, shape (N, 2)
        lnprob : ndarray, shape (N,)
            Initialized to -inf.
        outbounds : ndarray of bool, shape (N,)
        """
        ex = samp[:, 0]
        ey = samp[:, 1]
        r_sq = ex ** 2 + ey ** 2
        in_disk = r_sq <= self.ecc_max ** 2

        if self.flat_in_eccentricity:
            # p = 1/(2*pi*ecc_max * r)  =>  ln p = _log_norm - 0.5*ln(r^2)
            mask = in_disk & ~outbounds & (r_sq > 0)
            lnprob[mask] = self._log_norm - 0.5 * np.log(r_sq[mask])
        else:
            # p = 1/(pi*ecc_max^2)  =>  ln p = _log_norm
            mask = in_disk & ~outbounds
            lnprob[mask] = self._log_norm

        return lnprob


# ---------------------------------------------------------------------------
# Per-parameter JointPrior wrapper
# ---------------------------------------------------------------------------

class EccentricityVectorPrior(JointPrior):
    """Single-parameter prior for one component of the eccentricity vector.

    Parameters
    ----------
    dist : EccentricityVectorDist
        The shared joint distribution.
    name : str
        Must be one of `dist.names`.
    latex_label : str, optional
    unit : str, optional
    """

    def __init__(self, dist, name=None, latex_label=None, unit=None):
        if not isinstance(dist, EccentricityVectorDist):
            raise JointPriorDistError(
                "dist must be an instance of EccentricityVectorDist"
            )
        super().__init__(dist=dist, name=name, latex_label=latex_label, unit=unit)


# ---------------------------------------------------------------------------
# Waveform generator
# ---------------------------------------------------------------------------

class CartesianEccWaveformGenerator(GWSignalWaveformGenerator):
    """GWSignalWaveformGenerator that accepts (ecc_x, ecc_y) as sampled
    parameters and converts them to (eccentricity, mean_per_ano) before
    calling the waveform model.

    The disk boundary is enforced by the prior (`EccentricityVectorDist`
    rescale + `Constraint` on eccentricity), so no likelihood-side
    rejection is needed here.
    """

    def __init__(self, **kwargs):
        # Wrap whatever conversion function was requested (or the default)
        original_conversion = kwargs.pop(
            "parameter_conversion", convert_to_lal_binary_black_hole_parameters
        )

        def _convert_with_cartesian_ecc(parameters):
            _add_eccentricity_from_cartesian(parameters)
            return original_conversion(parameters)

        kwargs["parameter_conversion"] = _convert_with_cartesian_ecc
        super().__init__(**kwargs)

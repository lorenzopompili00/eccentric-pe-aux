"""
Bilby auxiliary tools for eccentric PE.

Provides Cartesian eccentricity vector parameterization for bilby,
a hyperbolic waveform generator, and an astrophysically motivated
single-single GW-capture prior on the generic-orbit initial conditions.
"""

from .ecc_cartesian import (
    EccentricityVectorDist,
    EccentricityVectorPrior,
    CartesianEccWaveformGenerator,
    convert_to_cartesian_ecc_bbh_parameters,
    generate_all_cartesian_ecc_bbh_parameters,
)
from .hyperbolic import HyperbolicGWSignalWaveformGenerator
from .capture_prior import (
    CaptureEnergyPrior,
    CaptureMomentumPrior,
    make_capture_prior_dict,
    convert_to_capture_bbh_parameters,
    generate_all_capture_bbh_parameters,
    CaptureHyperbolicWaveformGenerator,
)

__all__ = [
    "EccentricityVectorDist",
    "EccentricityVectorPrior",
    "CartesianEccWaveformGenerator",
    "convert_to_cartesian_ecc_bbh_parameters",
    "generate_all_cartesian_ecc_bbh_parameters",
    "HyperbolicGWSignalWaveformGenerator",
    "CaptureEnergyPrior",
    "CaptureMomentumPrior",
    "make_capture_prior_dict",
    "convert_to_capture_bbh_parameters",
    "generate_all_capture_bbh_parameters",
    "CaptureHyperbolicWaveformGenerator",
]

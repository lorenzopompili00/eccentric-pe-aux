"""
Hyperbolic waveform generator for bilby PE.

Extends :class:`GWSignalWaveformGenerator` with a ``hyperbolic`` flag that
adds ``energy`` and ``momentum`` as sampled parameters.

Usage in config.ini
-------------------
    waveform-generator = eccentric_pe_aux.bilby_aux.hyperbolic.HyperbolicGWSignalWaveformGenerator
"""

from bilby.gw.waveform_generator import GWSignalWaveformGenerator


class HyperbolicGWSignalWaveformGenerator(GWSignalWaveformGenerator):
    """GWSignalWaveformGenerator with an additional ``hyperbolic`` setting.

    When ``hyperbolic=True`` the parameters ``energy`` and ``momentum`` are
    included in the sampled parameter set.

    TODO: in this case the waveform is not initialized at a given frequency,
    but at a given separation. To be implemented.
    """

    _hyperbolic_parameters = {
        "energy",
        "momentum",
    }

    _all_parameters = GWSignalWaveformGenerator._all_parameters | _hyperbolic_parameters

    def __init__(self, hyperbolic=True, **kwargs):
        self.hyperbolic = hyperbolic
        super().__init__(**kwargs)

    @property
    def meta_data(self):
        md = super().meta_data
        md["hyperbolic"] = self.hyperbolic
        return md

    @property
    def defaults(self):
        defaults = super().defaults
        if not self.hyperbolic:
            defaults.update({p: 0.0 for p in self._hyperbolic_parameters})
        return defaults

    def _parameters_from_source_model(self):
        return super()._parameters_from_source_model() | self._hyperbolic_parameters

    def _from_bilby_parameters(self, **parameters):
        gwsignal_dict = super()._from_bilby_parameters(**parameters)
        gwsignal_dict["energy"] = parameters["energy"]
        gwsignal_dict["momentum"] = parameters["momentum"]
        return gwsignal_dict

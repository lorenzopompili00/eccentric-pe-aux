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

    When ``hyperbolic=True`` the parameters ``energy``, ``momentum``, and ``separation``
    are included in the sampled parameter set.
    """

    _hyperbolic_parameters = {
        "energy",
        "momentum",
        "separation",
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

    def _create_generator(self, waveform_approximant=None):
        if waveform_approximant is None:
            waveform_approximant = self.waveform_approximant

        # Use local pyseobnr plugin instead of the one in lalsimulation
        _hyperbolic_approxs = {"SEOBNRv6EHM"}
        if waveform_approximant in _hyperbolic_approxs:
            from .gwsignal_plugin import SEOBNRv6EHM
            _cls = {"SEOBNRv6EHM": SEOBNRv6EHM}
            return _cls[waveform_approximant]()

        try:
            from lalsimulation.gwsignal import gwsignal_get_waveform_generator
        except ImportError:
            raise ImportError("lalsimulation is not installed. Cannot use the GWSignal waveform generator.")
        return gwsignal_get_waveform_generator(waveform_approximant)

    def _from_bilby_parameters(self, **parameters):
        gwsignal_dict = super()._from_bilby_parameters(**parameters)
        gwsignal_dict["energy"] = parameters["energy"]
        gwsignal_dict["momentum"] = parameters["momentum"]
        gwsignal_dict["separation"] = parameters["separation"]
        return gwsignal_dict

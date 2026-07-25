"""Direct-NR waveform generator for bilby PE.

Runs parameter estimation with an SXS numerical-relativity waveform as the
template.  The intrinsic parameters (mass ratio, spins, eccentricity) are fixed
by the NR simulation, so only the total mass and the extrinsic parameters are
sampled.

The waveform matches the NR injection recipe in
``NR_injections/make_injection_NR.py``: the SXS modes are conditioned once
(tapered at the start, rolled off after the ringdown), summed over spin-weighted
spherical harmonics for the requested viewing angle, and rescaled from geometric
units to seconds and strain.

Turning the time-domain waveform into a frequency-domain one is left to
gwsignal's conditioning (a start taper followed by an FFT) -- the same
path bilby uses for other time-domain models, so results are directly
comparable.  For the details of that conditioning see the TEOBResumS-DALI merge
request: https://git.ligo.org/lscsoft/lalsuite/-/merge_requests/2496.

Usage in config.ini
--------------------
    waveform-generator = eccentric_pe_aux.bilby_aux.nr_sxs.NRGWSignalWaveformGenerator
    waveform-generator-constructor-dict = {'sxs_id': 'SXS:BBH:0001'}
    # or: {'sim_file': '/path/to/Strain_N2.h5'}
    waveform_approximant = NR_SXS

"""

import numpy as np
import lal
import astropy.units as u
from gwpy.timeseries import TimeSeries
from scipy.interpolate import CubicSpline

from lalsimulation.gwsignal.core.waveform import CompactBinaryCoalescenceGenerator
from lalsimulation.gwsignal.core.gw import GravitationalWaveModes
from bilby.gw.waveform_generator import GWSignalWaveformGenerator

#: approximant string that selects the NR generator
NR_APPROXIMANT = "NR_SXS"


def _value(x, unit):
    """Return the plain value of ``x`` in ``unit`` (accepts a Quantity or a float)."""
    if isinstance(x, u.Quantity):
        return x.to_value(unit)
    return float(x)


class NRWaveformGenerator(CompactBinaryCoalescenceGenerator):
    """gwsignal generator returning a fixed SXS-NR waveform.

    The SXS modes are loaded and conditioned once at construction (the intrinsic
    parameters never change).  Each call only sums the modes for the requested
    viewing angle, rescales by total mass and distance, and resamples onto the
    requested time step.

    Times are given in units of M (geometric units, G = c = 1), which physical
    total mass then converts to seconds.

    Parameters
    ----------
    sxs_id : str, optional
        SXS catalog identifier (e.g. ``"SXS:BBH:0001"``).  Loaded with
        ``sxs.load(sxs_id).h``.  Give exactly one of ``sxs_id`` / ``sim_file``.
    sim_file : str, optional
        Path to an SXS strain file, loaded with ``sxs.load(sim_file)``.
    t_taper : float, optional
        Length (in M) of the taper applied at the start of the waveform to bring
        it smoothly up from zero (default 500).
    ringdown_keep_M : float, optional
        Keep the waveform only up to ``peak + ringdown_keep_M`` (in M; default
        200).  The SXS conditioning appends a long flat tail after the ringdown;
        it must be cut off, otherwise the frequency-domain step (which keeps the
        last chunk of the waveform) would land on that tail rather than the
        merger.
    reference_time : float, optional
        NR reference time (in M).  Only needed for strain files that do not
        carry it in their metadata; otherwise it is read automatically.
    ell_max : int, optional
        Keep only modes with ``ell <= ell_max`` (default 5).  Higher-ell modes
        carry little power; dropping them speeds up generation for a negligible
        change in accuracy.  Raise it (e.g. 8) to use every mode the simulation
        provides.
    """

    def __init__(
        self,
        sxs_id=None,
        sim_file=None,
        t_taper=500.0,
        ringdown_keep_M=200.0,
        reference_time=None,
        ell_max=5,
        **kwargs,
    ):
        super().__init__()
        if (sxs_id is None) == (sim_file is None):
            raise ValueError("Provide exactly one of 'sxs_id' or 'sim_file'.")
        self.sxs_id = sxs_id
        self.sim_file = sim_file
        self.t_taper = float(t_taper)
        self.ringdown_keep_M = float(ringdown_keep_M)
        self.reference_time = reference_time
        self.ell_max = int(ell_max)

        self._implemented_domain = "time"
        self._generation_domain = None
        self._load_and_preprocess()

    @property
    def metadata(self):
        # These flags tell gwsignal to treat this as a time-domain model and do
        # the time-to-frequency conditioning itself; with f_ref_spin/f_ref_ecc
        # set to False it uses the "taper the start only" variant.
        return {
            "type": "nr",
            "f_ref_spin": False,
            "f_ref_ecc": False,
            "modes": True,
            "polarizations": True,
            "implemented_domain": "time",
            "approximant": NR_APPROXIMANT,
            "implementation": "",
            "conditioning_routines": "gwsignal",
        }

    def _load_and_preprocess(self):
        """Load the SXS modes and cache the geometric, trimmed mode arrays."""
        import sxs

        if self.sim_file is not None:
            wf = sxs.load(self.sim_file)
            w = wf
        else:
            wf = sxs.load(self.sxs_id)
            w = wf.h

        ref = self.reference_time
        if ref is None:
            try:
                ref = wf.metadata.reference_time
            except AttributeError:
                raise ValueError(
                    "Could not read reference_time from the SXS file metadata; "
                    "pass reference_time explicitly."
                )
        self.reference_time = ref

        w = w[w.index_closest_to(ref) :, :]
        w = w.preprocess(t1=ref, t2=ref + self.t_taper)

        peak = w.max_norm_time()
        t_geo = np.asarray(w.t) - peak
        # Keep from the taper start (the reference time) to peak+ringdown_keep_M,
        # dropping the flat padding the SXS conditioning adds before the start
        # and after the ringdown.
        keep = (t_geo >= ref - peak) & (t_geo <= self.ringdown_keep_M)

        self._t_geo = t_geo[keep]
        self._modes_geo = {
            (int(l), int(m)): np.asarray(w[:, w.index(l, m)])[keep]
            for (l, m) in w.LM
            if int(l) <= self.ell_max
        }
        if not self._modes_geo:
            raise ValueError(
                f"No SXS modes with ell <= {self.ell_max}; use ell_max >= 2."
            )
        self._build_splines()

    def _build_splines(self):
        """Build one cubic spline over all modes, once.

        total_mass only rescales the time axis (``t_phys = fac_t * t_geo``), so
        the spline is built over the fixed geometric grid and merely evaluated
        per call.  All modes are stacked (real then imaginary parts) into a
        single spline so one vectorised call resamples every mode at once.
        """
        self._lm = list(self._modes_geo.keys())
        cols = [self._modes_geo[lm].real for lm in self._lm]
        cols += [self._modes_geo[lm].imag for lm in self._lm]
        self._spline = CubicSpline(self._t_geo, np.column_stack(cols))

    def _eval_modes(self, parameters):
        """Resample all geometric modes onto the physical grid.

        Returns ``(t_uniform, deltaT, fac_h, modes)`` where ``modes`` is a
        complex ``(n_times, n_modes)`` array ordered as ``self._lm``.
        """
        mtot = _value(parameters["mass1"], u.solMass) + _value(
            parameters["mass2"], u.solMass
        )
        dl_m = _value(parameters["distance"], u.m)
        deltaT = _value(parameters["deltaT"], u.s)

        fac_t = mtot * lal.MTSUN_SI
        fac_h = -mtot * lal.MRSUN_SI / dl_m  # matches make_injection_NR.py

        # physical uniform grid; query the fixed geometric spline at t / fac_t
        t_uniform = np.arange(self._t_geo[0] * fac_t, self._t_geo[-1] * fac_t, deltaT)
        vals = self._spline(t_uniform / fac_t)  # (n_times, 2 * n_modes)
        k = len(self._lm)
        modes = vals[:, :k] + 1j * vals[:, k:]  # geometric, no fac_h yet
        return t_uniform, deltaT, fac_h, modes

    def generate_td_modes(self, **parameters):
        t_uniform, deltaT, fac_h, modes = self._eval_modes(parameters)
        t0 = float(t_uniform[0])
        return GravitationalWaveModes(
            {
                (l, m): TimeSeries(
                    modes[:, k] * fac_h, dt=deltaT, t0=t0, name=f"h_{l}_{m}"
                )
                for k, (l, m) in enumerate(self._lm)
            }
        )

    def generate_td_waveform(self, **parameters):
        # Viewing angles, following make_injection_NR.py: the polar angle is the
        # inclination and the azimuth is pi/2 - phase.
        theta = _value(parameters["inclination"], u.rad)
        phi = np.pi / 2.0 - _value(parameters["phi_ref"], u.rad)

        t_uniform, deltaT, fac_h, modes = self._eval_modes(parameters)
        # Combine the modes into the two polarizations: h+ - i hx = sum_lm Y_lm h_lm,
        # with Y_lm the spin-weight -2 spherical harmonics for this viewing angle.
        ylm = np.array(
            [
                lal.SpinWeightedSphericalHarmonic(theta, phi, -2, l, m)
                for (l, m) in self._lm
            ]
        )
        hpc = (modes @ ylm) * fac_h  # (n_times,)
        t0 = float(t_uniform[0])
        return (
            TimeSeries(hpc.real, dt=deltaT, t0=t0, name="hplus"),
            TimeSeries(-hpc.imag, dt=deltaT, t0=t0, name="hcross"),
        )

    # Pickle only the cached arrays and construction arguments, so the loaded NR
    # data is sent to the sampling processes once instead of reloaded by each.
    def __getstate__(self):
        return {
            "sxs_id": self.sxs_id,
            "sim_file": self.sim_file,
            "t_taper": self.t_taper,
            "ringdown_keep_M": self.ringdown_keep_M,
            "reference_time": self.reference_time,
            "ell_max": self.ell_max,
            "_implemented_domain": self._implemented_domain,
            "_generation_domain": self._generation_domain,
            "_t_geo": self._t_geo,
            "_modes_geo": self._modes_geo,
        }

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Rebuild the spline after unpickling; it is not stored, to keep the
        # pickled object small.
        self._build_splines()


class NRGWSignalWaveformGenerator(GWSignalWaveformGenerator):
    """bilby waveform generator that uses an SXS-NR waveform as the template.

    Sample the total mass and extrinsic parameters only; the NR intrinsic
    parameters are fixed.  Spins are fixed by the simulation and never sampled,
    so ``spinning`` is always False and the sampled ``theta_jn`` maps directly to
    the viewing angle ``iota``.

    The ``sxs_id`` / ``sim_file`` / ``t_taper`` / ``ringdown_keep_M`` /
    ``reference_time`` / ``ell_max`` arguments may be given either in
    ``waveform-generator-constructor-dict`` (preferred) or in
    ``waveform-arguments-dict``.  See :class:`NRWaveformGenerator` for their
    meaning and default values.
    """

    #: pickle the (already-loaded) NR generator to the sampling processes
    generator_pickles = True

    #: passed through to NRWaveformGenerator, with its default when unset
    _nr_defaults = dict(
        sxs_id=None,
        sim_file=None,
        t_taper=500.0,
        ringdown_keep_M=200.0,
        reference_time=None,
        ell_max=5,
    )

    def __init__(self, **kwargs):
        # NR spins are fixed by the simulation and never sampled, so spinning is
        # always False and is not a user option (ignore any stray value).
        kwargs.pop("spinning", None)
        # pull out the NR-specific arguments before bilby sees the rest; keep
        # only the ones actually given, so unset ones fall back below.  Stash
        # before super().__init__, which calls _create_generator().
        self._nr_kwargs = {k: kwargs.pop(k) for k in self._nr_defaults if k in kwargs}
        super().__init__(spinning=False, **kwargs)

    def _create_generator(self, waveform_approximant=None):
        if waveform_approximant is None:
            waveform_approximant = self.waveform_approximant
        if waveform_approximant == NR_APPROXIMANT:
            # resolve each argument: explicit constructor value, else the config's
            # waveform-arguments-dict, else the NRWaveformGenerator default.
            wa = self.waveform_arguments
            resolved = {
                key: self._nr_kwargs.get(key, wa.get(key, default))
                for key, default in self._nr_defaults.items()
            }
            return NRWaveformGenerator(**resolved)
        return super()._create_generator(waveform_approximant)

    def _from_bilby_parameters(self, **parameters):
        gwsignal_dict = super()._from_bilby_parameters(**parameters)
        # drop construction-only extras that may have leaked in via
        # waveform-arguments-dict (harmless, but keeps the gwsignal dict clean)
        for key in (
            "sxs_id",
            "sim_file",
            "t_taper",
            "ringdown_keep_M",
            "reference_time",
            "ell_max",
        ):
            gwsignal_dict.pop(key, None)
        return gwsignal_dict

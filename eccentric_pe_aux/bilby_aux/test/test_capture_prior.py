"""Sanity checks for the capture_prior module.

Verifies that the astrophysical single-single capture prior can be built as a 
ConditionalPriorDict, sampled, evaluated, and that it reproduces the underlying
forward model, with nu taken from the sampled mass_ratio and sigma a sampled
parameter.
"""
import sys
sys.path.insert(0, ".")
import numpy as np
import bilby
from bilby.core.prior import ConditionalPriorDict, Uniform

import eccentric_pe_aux.bilby_aux.capture_prior as cap

C_KMS = cap.C_KMS
PK = cap.PETERS_K

# =====================================================================
# 1. Build via make_capture_prior_dict and draw samples
# =====================================================================
priors = cap.make_capture_prior_dict()          # default sigma ~ U(0.1, 1000)
assert isinstance(priors, ConditionalPriorDict)
print("prior keys:", list(priors.keys()))
print("sampling order:", priors.sorted_keys)

bilby.core.utils.random.seed(123)
N = 200_000
s = priors.sample(N)
q   = np.asarray(s["mass_ratio"])
nu  = cap.nu_of_q(q)
sig = np.asarray(s["sigma"])
dE  = np.asarray(s["delta_energy"])     # = E_0 - 1
pph = np.asarray(s["momentum"])

print("\nConditionalPriorDict sample (nu from mass_ratio, sigma sampled):")
for nm, a in [("sigma", sig), ("nu", nu), ("delta_energy", dE), ("momentum", pph)]:
    lo, md, hi = np.percentile(a, [5, 50, 95])
    print(f"  {nm:13s} median {md:.4g}  90% CI [{lo:.4g}, {hi:.4g}]")

assert np.all(dE > 0.0), "delta_energy must be > 0 (Gamma support)"
assert np.all(pph >= 0.0), "momentum must be >= 0"
assert np.all(np.isfinite(dE)) and np.all(np.isfinite(pph))

# =====================================================================
# 2. Cross-check against the direct forward model on the SAME (nu, sigma)
#    draws.  delta_energy = (2 nu sigma^2/c^2) u, u ~ Gamma(5/7, 1);
#    pphi = sqrt(2 rp_hat), rp_hat ~ U(0, rp_max_hat).
# =====================================================================
rng = np.random.default_rng(123)
u    = rng.gamma(5.0 / 7.0, 1.0, N)
dE2  = 2.0 * nu * (sig / C_KMS) ** 2 * u
b2   = 2.0 * dE2 / nu
rpmx = (PK * nu / b2) ** (2.0 / 7.0)
pph2 = np.sqrt(2.0 * rng.uniform(0, 1, N) * rpmx)

print("\nDirect forward model (same nu, sigma draws):")
for nm, a in [("delta_energy", dE2), ("momentum", pph2)]:
    lo, md, hi = np.percentile(a, [5, 50, 95])
    print(f"  {nm:13s} median {md:.4g}  90% CI [{lo:.4g}, {hi:.4g}]")

# medians should agree to within a few % (independent RNG streams / ordering)
for name, a, b in [("delta_energy", dE, dE2), ("momentum", pph, pph2)]:
    r = np.median(a) / np.median(b)
    print(f"  median ratio {name}: {r:.3f}")
    assert 0.9 < r < 1.1, f"{name} forward-model mismatch (ratio {r:.3f})"

# =====================================================================
# 3. ln_prob finite at sampled points (sampler needs this)
# =====================================================================
lp = np.array([priors.ln_prob({k: s[k][i] for k in s})
               for i in range(0, N, N // 10)])
print("\nln_prob at sampled points: all finite =", bool(np.all(np.isfinite(lp))))
assert np.all(np.isfinite(lp))

# =====================================================================
# 4. rescale (dynesty prior_transform) works through the conditional chain
# =====================================================================
keys = [k for k in priors.sorted_keys if k not in ("mass_1", "mass_2")]
print("transform keys:", keys)
rngc = np.random.default_rng(0)
for _ in range(5):
    cube = rngc.uniform(0, 1, len(keys))
    vals = priors.rescale(keys, cube)
    d = {k: float(np.atleast_1d(v)[0]) for k, v in zip(keys, vals)}
    lpc = priors.ln_prob(d)
    assert np.isfinite(lpc), f"non-finite ln_prob from rescale: {d}"
print("rescale -> finite ln_prob for 5 unit-cube draws: OK")

# =====================================================================
# 5. Custom sigma prior is honoured
# =====================================================================
pr2 = cap.make_capture_prior_dict(
    sigma_prior=Uniform(name="sigma", minimum=5.0, maximum=20.0))
s2 = pr2.sample(50_000)
sg2 = np.asarray(s2["sigma"])
print(f"\ncustom sigma U(5,20): min={sg2.min():.2f}, max={sg2.max():.2f}")
assert sg2.min() >= 5.0 - 1e-6 and sg2.max() <= 20.0 + 1e-6

# =====================================================================
# 6. Waveform-generator conversion adds energy = 1 + delta_energy
# =====================================================================
samp_wf = {"chirp_mass": 80.0, "mass_ratio": 0.8,
           "delta_energy": 0.05, "momentum": 4.0, "sigma": 100.0}
conv, added = cap.convert_to_capture_bbh_parameters(samp_wf.copy())
print(f"\nconvert_to_capture_bbh_parameters: energy = {conv['energy']:.4f} "
      f"(expected 1.05), added_keys = {added}")
assert abs(conv["energy"] - 1.05) < 1e-12
assert "energy" in added

# =====================================================================
# 7. Standalone priors importable & repr round-trips (prior-file usage)
# =====================================================================
de = cap.CaptureEnergyPrior(name="delta_energy")
mo = cap.CaptureMomentumPrior(name="momentum")
print("\nrepr(CaptureEnergyPrior):  ", repr(de))
print("repr(CaptureMomentumPrior):", repr(mo))

print("\nAll checks passed.")

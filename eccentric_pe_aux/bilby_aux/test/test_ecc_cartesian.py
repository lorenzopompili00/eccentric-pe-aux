"""Quick sanity checks for ecc_cartesian module."""
import sys
sys.path.insert(0, ".")
import numpy as np
import eccentric_pe_aux.bilby_aux.ecc_cartesian as ec

# =====================================================================
# 1. EccentricityVectorDist basics
# =====================================================================
dist = ec.EccentricityVectorDist(names=["ecc_x", "ecc_y"], ecc_max=0.4)
print("distname:", dist.distname)
print("bounds:", dist.bounds)

# Rescale: unit cube -> disk
u = np.array([[0.5, 0.0], [0.5, 0.25], [1.0, 0.0], [0.0, 0.0]])
rescaled = dist.rescale(u)
print("\nrescaled from unit cube:")
for i, row in enumerate(np.atleast_2d(rescaled)):
    e = np.sqrt(row[0] ** 2 + row[1] ** 2)
    print(f"  u={u[i]} -> ({row[0]:.4f}, {row[1]:.4f}), e={e:.4f}")

# Sample statistics (default is flat_in_eccentricity=True => E[e] = ecc_max/2)
samples = dist._sample(10000)
e_samples = np.sqrt(samples[:, 0] ** 2 + samples[:, 1] ** 2)
print(f"\n10k samples: e_max={e_samples.max():.4f}, e_mean={e_samples.mean():.4f}")
print(f"  (expected mean for flat p(e) on [0,0.4]: {0.4/2:.4f})")
assert e_samples.max() <= 0.4 + 1e-10, "sample outside disk!"

# ln_prob: p(ex,ey) = 1/(2*pi*ecc_max*e) => ln p = -ln(2*pi*ecc_max) - 0.5*ln(ex^2+ey^2)
vals = np.array([[0.1, 0.1], [0.0, 0.5]])  # second is outside disk
lnp = dist.ln_prob(vals)
expected_lnp = -np.log(2 * np.pi * 0.4) - 0.5 * np.log(0.1**2 + 0.1**2)
print(f"\nln_prob inside disk:  {lnp[0]:.4f} (expected {expected_lnp:.4f})")
print(f"ln_prob outside disk: {lnp[1]:.4f} (expected -inf)")
assert np.isfinite(lnp[0])
assert np.abs(lnp[0] - expected_lnp) < 1e-10
assert lnp[1] == -np.inf

# =====================================================================
# 2. JointPrior rescale mechanism
# =====================================================================
prior_x = ec.EccentricityVectorPrior(dist=dist, name="ecc_x", latex_label=r"$e_x$")
prior_y = ec.EccentricityVectorPrior(dist=dist, name="ecc_y", latex_label=r"$e_y$")
print(f"\nprior_x bounds: [{prior_x.minimum}, {prior_x.maximum}]")

dist.reset_rescale()
r1 = prior_x.rescale(0.5)
print(f"partial rescale (ecc_x only): {r1}")  # should be []
r2 = prior_y.rescale(0.25)
print(f"full rescale: {r2}")  # should be the 2-element array
assert len(r1) == 0, "first rescale should return []"
assert len(r2) == 2, "second rescale should return 2 values"

# =====================================================================
# 3. uniform-on-disk mode (flat_in_eccentricity=False => p(e) ~ e)
# =====================================================================
dist2 = ec.EccentricityVectorDist(
    names=["ecc_x", "ecc_y"], ecc_max=0.4, flat_in_eccentricity=False
)
samples2 = dist2._sample(10000)
e2 = np.sqrt(samples2[:, 0] ** 2 + samples2[:, 1] ** 2)
print(f"\nuniform-on-disk: e_mean={e2.mean():.4f} (expected {2*0.4/3:.4f})")
assert e2.max() <= 0.4 + 1e-10

# =====================================================================
# 4. convert_to_cartesian_ecc_bbh_parameters (waveform-generator signature)
# =====================================================================
sample_wf = {"chirp_mass": 1.19, "mass_ratio": 0.9, "ecc_x": 0.2, "ecc_y": 0.1}
converted_wf, added_keys = ec.convert_to_cartesian_ecc_bbh_parameters(sample_wf.copy())
print(f"\nconvert_to_cartesian_ecc_bbh_parameters:")
print(f"  eccentricity = {converted_wf['eccentricity']:.4f}")
print(f"  mean_per_ano = {converted_wf['mean_per_ano']:.4f}")
print(f"  added_keys = {added_keys}")
assert "eccentricity" in converted_wf
assert "mean_per_ano" in converted_wf
assert "eccentricity" in added_keys
assert "mean_per_ano" in added_keys
assert isinstance(added_keys, list)

# =====================================================================
# 5. repr round-trip (needed for prior file parsing)
# =====================================================================
print(f"\nrepr(dist):    {repr(dist)}")
print(f"repr(prior_x): {repr(prior_x)}")

print("\nAll checks passed.")

"""Quick sanity checks for ecc_cartesian module."""
import sys
sys.path.insert(0, ".")
import numpy as np
import ecc_cartesian as ec

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

# Sample statistics
samples = dist._sample(10000)
e_samples = np.sqrt(samples[:, 0] ** 2 + samples[:, 1] ** 2)
print(f"\n10k samples: e_max={e_samples.max():.4f}, e_mean={e_samples.mean():.4f}")
print(f"  (expected mean for p(e)~e on [0,0.4]: {2*0.4/3:.4f})")
assert e_samples.max() <= 0.4 + 1e-10, "sample outside disk!"

# ln_prob
vals = np.array([[0.1, 0.1], [0.0, 0.5]])  # second is outside disk
lnp = dist.ln_prob(vals)
print(f"\nln_prob inside disk:  {lnp[0]:.4f} (expected {-np.log(np.pi*0.16):.4f})")
print(f"ln_prob outside disk: {lnp[1]:.4f} (expected -inf)")
assert np.isfinite(lnp[0])
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
# 3. flat_in_eccentricity mode
# =====================================================================
dist2 = ec.EccentricityVectorDist(
    names=["ecc_x", "ecc_y"], ecc_max=0.4, flat_in_eccentricity=True
)
samples2 = dist2._sample(10000)
e2 = np.sqrt(samples2[:, 0] ** 2 + samples2[:, 1] ** 2)
print(f"\nflat_in_ecc: e_mean={e2.mean():.4f} (expected {0.4/2:.4f})")
assert e2.max() <= 0.4 + 1e-10

# =====================================================================
# 4. bbh_ecc_conversion + Constraint
# =====================================================================
from bilby.core.prior import Constraint

sample_inside = {"chirp_mass": 1.19, "mass_ratio": 0.9, "ecc_x": 0.1, "ecc_y": 0.1}
out = ec.bbh_ecc_conversion(sample_inside.copy())
print(f"\nbbh_ecc_conversion:")
print(f"  eccentricity = {out['eccentricity']:.4f}")
print(f"  mean_per_ano = {out['mean_per_ano']:.4f}")
print(f"  mass_1 = {out.get('mass_1', 'MISSING')}")
print(f"  mass_2 = {out.get('mass_2', 'MISSING')}")
assert "eccentricity" in out
assert "mass_1" in out

# Simulate evaluate_constraints behavior
ecc_constraint = Constraint(name="eccentricity", minimum=0, maximum=0.4)

sample_in = {"ecc_x": 0.1, "ecc_y": 0.2, "chirp_mass": 1.19, "mass_ratio": 0.9}
converted = ec.bbh_ecc_conversion(sample_in.copy())
print(f"\nConstraint test (e={converted['eccentricity']:.3f}, inside):")
print(f"  prob = {ecc_constraint.prob(converted['eccentricity'])}")

sample_out = {"ecc_x": 0.3, "ecc_y": 0.3, "chirp_mass": 1.19, "mass_ratio": 0.9}
converted_out = ec.bbh_ecc_conversion(sample_out.copy())
print(f"Constraint test (e={converted_out['eccentricity']:.3f}, outside):")
print(f"  prob = {ecc_constraint.prob(converted_out['eccentricity'])}")

# =====================================================================
# 5. repr round-trip (needed for prior file parsing)
# =====================================================================
print(f"\nrepr(dist):    {repr(dist)}")
print(f"repr(prior_x): {repr(prior_x)}")

print("\nAll checks passed.")

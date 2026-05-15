# Single-single GW-capture prior

This document describes the astrophysical prior implemented in `capture_prior.py` based on the single-single gravitational-wave capture forward model of Gupte et al. 2026 (arXiv:2603.29019).

A dynamically formed close binary can be approximated as a near-parabolic encounter. 
Reduce the two-body problem to an effective particle of reduced mass $\mu$ in the field of the total mass $M$ ($\nu=\mu/M$ the symmetric mass ratio). 
It comes in from infinity with relative speed $v_\infty$ and impact parameter $b$; the Newtonian conserved quantities (per unit reduced mass) are

$$\varepsilon = \tfrac{1}{2}v_\infty^2 \quad(\text{specific energy}), \qquad \ell = b\,v_\infty \quad(\text{specific angular momentum}).$$

$\varepsilon$ distinguishes bound from unbound and maps to $E_0-1$; $\ell$ fixes the pericenter distance $r_p$ and maps to $p_\phi^0$. 
So the two initial conditions $(E_0,p_\phi^0)$ correspond to the two encounter parameters $(v_\infty, r_p)$.

**Note**: everything is in geometric units ($G=c=1$), and the total mass cancels out of the prior entirely.

- $E_0-1=\tfrac12\nu(v_\infty/c)^2$ — energy normalized by total rest mass; no $M$.
- $p_\phi^0=\sqrt{2\hat r_p}$  where $\hat r_p \equiv r_p/M$
- The Peters maximum-capture pericenter, re-expressed in units of $M$,
  reduces its mass factor to $\mu^{2/7}M^{5/7}/M=(\mu/M)^{2/7}=\nu^{2/7}$.

So the dimensionless prior on $(E_0,p_\phi^0)$ depends **only** on $\nu$ and the velocity distribution ($v_\infty$, hence $\sigma$).


## 1. The $E_0$ prior

In our conventions, $E_0$ is the energy normalized by total mass; at infinity the potential vanishes, so $E_0 = 1 + T_\infty/(Mc^2)$ with
$T_\infty=\tfrac12\mu v_\infty^2$ the centre-of-mass kinetic energy. Hence

$$\boxed{\,E_0 - 1 \;\simeq\; \tfrac{1}{2}\,\nu\,(v_\infty/c)^2\,}\qquad (\text{leading non-relativistic order}).$$

(The exact relation is $E_0^2 = 1 + 2\nu(\gamma_\infty-1)$; the correction is $\mathcal{O}((v_\infty/c)^4)$ and negligible for cluster velocities.)

For two BHs drawn from a thermal host with 1-D velocity dispersion $\sigma$, the relative speed at infinity follows a Maxwellian with relative-velocity dispersion $\sqrt{2}\,\sigma$, weighted by the GW-capture rate — the capture cross-section $\sigma_{\rm cap}\propto v_\infty^{-18/7}$ times the encounter flux $\propto v_\infty$. The net distribution is (see Eq. 4 of arXiv:2603.29019)

$$p(v_\infty\mid\sigma)\;\propto\; v_\infty^{3/7}\,
  \exp\!\left(-\frac{v_\infty^2}{4\sigma^2}\right).$$

Substitute $u = v_\infty^2/(4\sigma^2)$. Then $v_\infty^{3/7}e^{-u}\,dv_\infty
\propto u^{-2/7}e^{-u}\,du$, i.e. $u\sim\mathrm{Gamma}(\text{shape}=5/7,
\text{scale}=1)$. Combining with the $E_0\leftrightarrow v_\infty$ map,
$E_0-1=\tfrac12\nu v_\infty^2/c^2 = (2\nu\sigma^2/c^2)\,u$, so

$$\boxed{\,E_0 - 1 \;\sim\; \mathrm{Gamma}\!\left(k=\tfrac{5}{7},\;
  \theta = \tfrac{2\nu\sigma^2}{c^2}\right)\,}.$$


## 2. The $p_\phi^0$ prior

At pericenter $\dot r=0$, so energy conservation gives
$\tfrac12 v_\infty^2 = \ell^2/(2r_p^2) - GM/r_p$.

In the "gravitational focusing" regime, $\tfrac12 v_\infty^2 \ll GM/r_p$, one can drop the kinetic term

$$0\approx\frac{\ell^2}{2r_p^2}-\frac{GM}{r_p}
  \;\Longrightarrow\; \ell^2 = 2\,GM\,r_p.$$

In the EOB convention $p_\phi^0 = P_\phi^0/(\mu M)=\ell/M$ ($G=c=1$):

$$\boxed{\,p_\phi^0 \;=\; \frac{\ell}{M} \;=\;
  \sqrt{\frac{2\,r_p}{M}} \;=\; \sqrt{2\,\hat r_p}\,},\qquad
  \hat r_p\equiv r_p/M.$$

### Distribution of $r_p$ (Peters capture criterion)

In the focusing regime the cross-section to reach pericenter $\le r_p$ is $\propto r_p$, so $r_p$ is uniform up to a maximum set by requiring the GW energy radiated in one passage to exceed the kinetic energy at infinity, $\Delta E_{\rm GW}\ge\tfrac12\mu v_\infty^2$ (Peters 1964,
$\Delta E_{\rm GW}=\tfrac{85\pi}{12\sqrt2}G^{7/2}\mu^2M^{5/2}/(c^5 r_p^{7/2})$; see also arXiv:2603.29019).

Solving and expressing in units of $M$ (the mass factor collapses to $\nu^{2/7}$):

$$\hat r_{p,\max}=\left[\;\underbrace{\frac{85\pi}{6\sqrt2}}_{\texttt{PETERS\_K}}\; \frac{\nu}{\beta^2}\right]^{2/7},\qquad \beta^2=2(E_0-1)/\nu.$$

`PETERS_K` $=85\pi/(6\sqrt2)\approx31.47$.
Mapping $r_p$-uniform through $p_\phi^0=\sqrt{2\hat r_p}$ gives a density
$\propto p_\phi^0$, i.e. a power law of index 1:

$$\boxed{\,p(p_\phi^0\mid E_0,\nu)\;=\;\mathrm{PowerLaw}(\alpha=1)\ \text{on}\
  [0,\ \sqrt{2\,\hat r_{p,\max}}]\,}.$$

Note $r_p$ (hence $p_\phi^0$) is not independent of $E_0$: $\hat r_{p,\max}$ depends on $\beta^2\propto(E_0-1)$.

---

## 3. The joint prior as a conditional chain

The result is a joint prior $p(E_0,p_\phi^0\mid\nu,\sigma)$ — the two are correlated — but it is decomposable into the closed-form 1-D conditionals above:

$$p(E_0,p_\phi^0\mid\nu,\sigma)=
  \underbrace{p(E_0-1\mid\nu,\sigma)}_{\Gamma(5/7,\,2\nu\sigma^2/c^2)}\;
  \underbrace{p(p_\phi^0\mid E_0,\nu)}_{\mathrm{PowerLaw}(\alpha=1)}.$$

With $\nu$ from a sampled `mass_ratio` and $\sigma$ a sampled
parameter, the full prior chain is

```
sigma         ~  (user prior, e.g. Uniform(1, 1000) km/s)
mass_ratio    ~  (user prior)                      ->  nu = q/(1+q)^2
delta_energy  ~  Gamma(k = 5/7, theta = 2 nu sigma^2/c^2)        | nu, sigma
momentum      ~  PowerLaw(alpha = 1) on [min, sqrt(2 r_p,max_hat)] | E_0, nu
```

`delta_energy` $\equiv E_0-1$ is the sampled quantity; the waveform `energy` $= 1 + $ `delta_energy`.

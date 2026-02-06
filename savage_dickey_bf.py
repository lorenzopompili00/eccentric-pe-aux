import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

import bilby
import numpy as np
import matplotlib.pyplot as plt

from pesummary.utils.bounded_1d_kde import bounded_1d_kde


def uniform_prior_eccentricity(e, e_max=0.8):
    "Default prior: uniform in eccentricity up to 0.8"
    if 0 <= e <= e_max:
        return 1.0 / e_max
    else:
        return 0.0


def savage_dickey_bf(posterior_samples, prior_samples=None, prior_func=uniform_prior_eccentricity, 
                     test_value=0.0, kde_method='Reflection', plot=False):
    """
    Calculate Bayes Factor using the Savage-Dickey density ratio.
    
    The Savage-Dickey density ratio for testing H0: θ = θ0 vs H1: θ ≠ θ0 is:
    BF_{10} = p(θ = θ0) / p(θ = θ0 | data) 
    
    Parameters
    ----------
    posterior_samples : array-like
        Samples from the posterior distribution of the parameter
    prior_samples : array-like, optional
        Samples from the prior distribution. Either prior_samples or prior_func must be provided.
    prior_func : callable, optional
        Function that returns the prior density at a given value.
        Either prior_samples or prior_func must be provided.
    test_value : float, default=0.0
        The value to test (θ0). For eccentricity, this is typically 0.
    kde_method : str, default='Reflection'
        Method for handling the boundary in `bounded_1d_kde`
    plot : bool, default=False
        If True, plots the prior and posterior densities
        
    Returns
    -------
    log_10_bf : float
        Log 10 Bayes factor in favor of H1 (θ ≠ θ0). 
        log_10_bf > 0 means evidence for H1 (e.g., eccentric orbit)
    """
    posterior_samples = np.asarray(posterior_samples)
    
    kde_posterior = bounded_1d_kde(posterior_samples, method=kde_method)
    posterior_density = kde_posterior(test_value)[0]

    # Estimate prior density at test_value
    if prior_func is not None:
        prior_density = prior_func(test_value)
    elif prior_samples is not None:
        prior_samples = np.asarray(prior_samples)
        kde_prior = bounded_1d_kde(prior_samples, bw_method=kde_method)
        prior_density = kde_prior(test_value)[0]
    else:
        raise ValueError("Either prior_samples or prior_func must be provided")
    
    # Calculate Bayes Factor
    bf = prior_density / posterior_density
    log_10_bf = np.log10(bf)

    # Optional plotting
    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Create evaluation grid
        x_min = min(0, posterior_samples.min())
        x_max = posterior_samples.max()
        x_grid = np.linspace(x_min, x_max, 1000)
        
        # Evaluate densities
        kde_post_plot = bounded_1d_kde(posterior_samples, method=kde_method)
        posterior_densities = kde_post_plot(x_grid)
        
        if prior_func is not None:
            prior_densities = np.array([prior_func(x) for x in x_grid])
        else:
            kde_prior_plot = bounded_1d_kde(prior_samples, method=kde_method)
            prior_densities = kde_prior_plot(x_grid)
    
        # Plot

        # Plot histogram of posterior samples (normalized)
        ax.hist(posterior_samples, bins=100, density=True, alpha=0.3,
                color='C1', label='Posterior (histogram)', edgecolor='black', linewidth=0.5)
        ax.plot(x_grid, posterior_densities, label='Posterior (KDE)', linewidth=2, alpha=0.7)
        ax.plot(x_grid, prior_densities, label='Prior', linewidth=2, alpha=0.7)
        
        ax.axvline(test_value, color='red', linestyle='--', linewidth=2)
        ax.axhline(prior_density, color='C0', linestyle=':', alpha=0.5)
        ax.axhline(posterior_density, color='C1', linestyle=':', alpha=0.5)
        
        ax.set_xlabel('Parameter value', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'Savage-Dickey Density Ratio: $\log_{{{10}}}$(BF) = {log_10_bf:.3f}', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        
    return log_10_bf

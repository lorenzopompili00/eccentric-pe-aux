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


def savage_dickey_bf(
    posterior_samples,
    prior_samples=None,
    prior_func=uniform_prior_eccentricity,
    test_value=0.0,
    kde_method="Reflection",
    bw_method=None,
    plot=False,
):
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

    kde_posterior = bounded_1d_kde(
        posterior_samples,
        method=kde_method,
        bw_method=bw_method,
        xlow=test_value,
    )
    posterior_density = kde_posterior(test_value)[0]

    # Estimate prior density at test_value
    if prior_func is not None:
        prior_density = prior_func(test_value)
    elif prior_samples is not None:
        prior_samples = np.asarray(prior_samples)
        kde_prior = bounded_1d_kde(
            prior_samples,
            method=kde_method,
            bw_method=bw_method,
            xlow=test_value,
        )
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
        kde_post_plot = bounded_1d_kde(
            posterior_samples,
            method=kde_method,
            bw_method=bw_method,
            xlow=test_value,
        )
        posterior_densities = kde_post_plot(x_grid)

        if prior_func is not None:
            prior_densities = np.array([prior_func(x) for x in x_grid])
        else:
            kde_prior_plot = bounded_1d_kde(
                prior_samples,
                method=kde_method,
                bw_method=bw_method,
                xlow=test_value,
            )
            prior_densities = kde_prior_plot(x_grid)

        # Plot

        # Plot histogram of posterior samples (normalized)
        ax.hist(
            posterior_samples,
            bins=100,
            density=True,
            alpha=0.3,
            color="C1",
            label="Posterior (histogram)",
            edgecolor="black",
            linewidth=0.5,
        )
        ax.plot(
            x_grid,
            posterior_densities,
            label="Posterior (KDE)",
            linewidth=2,
            alpha=0.7,
            color="C1",
        )
        ax.plot(
            x_grid, prior_densities, label="Prior", linewidth=2, alpha=0.7, color="C0"
        )

        ax.axvline(test_value, color="red", linestyle="--", linewidth=2)
        ax.axhline(prior_density, color="C0", linestyle=":", alpha=0.5)
        ax.axhline(posterior_density, color="C1", linestyle=":", alpha=0.5)

        ax.set_xlabel("Parameter value", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(
            f"Savage-Dickey Density Ratio: $\log_{{{10}}}$(BF) = {log_10_bf:.3f}",
            fontsize=14,
        )
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    return log_10_bf


def savage_dickey_bf_uncertainty(
    posterior_samples,
    prior_samples=None,
    prior_func=uniform_prior_eccentricity,
    test_value=0.0,
    kde_method="Reflection",
    bw_methods=None,
    n_bootstrap=100,
    confidence_level=0.90,
    plot=False,
    random_seed=42,
):
    """
    Calculate Bayes Factor using the Savage-Dickey density ratio with robustness checks.

    This function provides uncertainty estimates through:
    1. Bootstrap resampling to estimate confidence intervals
    2. Multiple KDE bandwidths

    Reports median and confidence intervals.

    Parameters
    ----------
    posterior_samples : array-like
        Samples from the posterior distribution of the parameter
    prior_samples : array-like, optional
        Samples from the prior distribution
    prior_func : callable, optional
        Function that returns the prior density at a given value
    test_value : float, default=0.0
        The value to test (θ0). For eccentricity, this is typically 0.
    kde_method : str, default='Reflection'
        Method for handling the boundary in `bounded_1d_kde`
    bw_methods : list of float or str, optional
        List of bandwidth methods to test. If None, uses multiple scales of Scott's rule.
        Examples: [0.5, 0.75, 1.0, 1.25, 1.5] or ['scott', 'silverman']
    n_bootstrap : int, default=100
        Number of bootstrap samples
    confidence_level : float, default=0.90
        Confidence level for intervals (0 to 1)
    plot : bool, default=False
        If True, plots the results with uncertainty bands
    random_seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    result : dict
        Dictionary containing:
        - 'log10_bf_median': Median log10(BF)
        - 'log10_bf_mean': Mean log10(BF)
        - 'log10_bf_std': Standard deviation of log10(BF)
        - 'log10_bf_ci': Confidence interval [lower, upper]
        - 'bf_median': Median BF (in linear scale)
        - 'bf_ci': BF confidence interval [lower, upper]
        - 'all_log10_bfs': All bootstrap log10(BF) values
    """
    np.random.seed(random_seed)
    posterior_samples = np.asarray(posterior_samples)
    n_samples = len(posterior_samples)

    # Set default bandwidth methods if not provided
    if bw_methods is None:
        # Use multiple scales of the bandwidth
        bw_methods = [0.5, 0.75, 1.0, 1.25, 1.5]

    log10_bfs = []

    # Bootstrap over samples and bandwidth methods
    for _ in range(n_bootstrap):
        # Resample posterior
        bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        bootstrap_samples = posterior_samples[bootstrap_indices]

        # Try different bandwidth methods
        for bw_method in bw_methods:
            try:
                kde_posterior = bounded_1d_kde(
                    bootstrap_samples,
                    method=kde_method,
                    bw_method=bw_method,
                    xlow=test_value,
                )
                posterior_density = kde_posterior(test_value)[0]

                # Estimate prior density
                if prior_func is not None:
                    prior_density = prior_func(test_value)
                elif prior_samples is not None:
                    # Also bootstrap prior samples
                    prior_bootstrap_indices = np.random.choice(
                        len(prior_samples), size=len(prior_samples), replace=True
                    )
                    prior_bootstrap = prior_samples[prior_bootstrap_indices]
                    kde_prior = bounded_1d_kde(
                        prior_bootstrap,
                        method=kde_method,
                        bw_method=bw_method,
                        xlow=test_value,
                    )
                    prior_density = kde_prior(test_value)[0]

                # Calculate BF
                if posterior_density > 0:
                    bf = prior_density / posterior_density
                    log10_bfs.append(np.log10(bf))
            except:
                # Skip failed KDE attempts
                continue

    log10_bfs = np.array(log10_bfs)

    # Calculate statistics
    alpha = 1 - confidence_level
    log10_bf_median = np.median(log10_bfs)
    log10_bf_mean = np.mean(log10_bfs)
    log10_bf_std = np.std(log10_bfs)
    log10_bf_ci = np.percentile(log10_bfs, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    # Convert to linear scale
    bf_median = 10**log10_bf_median
    bf_ci = 10**log10_bf_ci

    result = {
        "log10_bf_median": log10_bf_median,
        "log10_bf_mean": log10_bf_mean,
        "log10_bf_std": log10_bf_std,
        "log10_bf_ci": log10_bf_ci,
        "bf_median": bf_median,
        "bf_ci": bf_ci,
        "all_log10_bfs": log10_bfs,
    }

    # Optional plotting
    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Distribution with KDE uncertainty
        x_min = min(0, posterior_samples.min())
        x_max = posterior_samples.max()
        x_grid = np.linspace(x_min, x_max, 1000)

        # Plot histogram
        ax1.hist(
            posterior_samples,
            bins=50,
            density=True,
            alpha=0.3,
            color="C1",
            label="Posterior (histogram)",
            edgecolor="black",
            linewidth=0.5,
        )

        # Plot KDE with uncertainty from different bandwidths
        kde_curves = []
        for bw_method in bw_methods:
            try:
                kde = bounded_1d_kde(
                    posterior_samples,
                    method=kde_method,
                    bw_method=bw_method,
                    xlow=test_value,
                )
                kde_curves.append(kde(x_grid))
            except:
                continue

        if kde_curves:
            kde_curves = np.array(kde_curves)
            kde_median = np.median(kde_curves, axis=0)
            kde_lower = np.percentile(kde_curves, 100 * alpha / 2, axis=0)
            kde_upper = np.percentile(kde_curves, 100 * (1 - alpha / 2), axis=0)

            ax1.plot(
                x_grid, kde_median, "C1", linewidth=2.5, label="Posterior (KDE median)"
            )
            ax1.fill_between(
                x_grid,
                kde_lower,
                kde_upper,
                color="C1",
                alpha=0.2,
                label=f"Posterior ({int(confidence_level*100)}% CI)",
            )

        # Plot prior
        if prior_func is not None:
            prior_densities = np.array([prior_func(x) for x in x_grid])
            ax1.plot(x_grid, prior_densities, "C0", linewidth=2.5, label="Prior")

        ax1.axvline(
            test_value,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Test value = {test_value}",
        )
        ax1.set_xlabel("Eccentricity", fontsize=12)
        ax1.set_ylabel("Density", fontsize=12)
        ax1.set_title("Posterior and Prior Distributions", fontsize=13)
        ax1.legend(fontsize=9)
        ax1.grid(alpha=0.3)

        # Plot 2: Bootstrap distribution of log10(BF)
        ax2.hist(
            log10_bfs,
            bins=30,
            density=True,
            alpha=0.6,
            color="purple",
            edgecolor="black",
        )
        ax2.axvline(
            log10_bf_median,
            color="red",
            linewidth=2.5,
            label=f"Median = {log10_bf_median:.2f}",
        )
        ax2.axvline(log10_bf_ci[0], color="red", linewidth=2, linestyle="--", alpha=0.7)
        ax2.axvline(log10_bf_ci[1], color="red", linewidth=2, linestyle="--", alpha=0.7)
        ax2.axvspan(
            log10_bf_ci[0],
            log10_bf_ci[1],
            alpha=0.2,
            color="red",
            label=f"{int(confidence_level*100)}% CI: [{log10_bf_ci[0]:.2f}, {log10_bf_ci[1]:.2f}]",
        )
        ax2.set_xlabel(r"$\log_{10}$(BF$_{10}$)", fontsize=12)
        ax2.set_ylabel("Density", fontsize=12)
        ax2.set_title(f"Bootstrap Distribution (N={len(log10_bfs)})", fontsize=13)
        ax2.legend(fontsize=10)
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

    return result

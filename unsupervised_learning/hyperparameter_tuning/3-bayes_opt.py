    #!/usr/bin/env python3
"""
Bayesian Optimization initialization module.
"""

import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    Performs Bayesian optimization on a noiseless 1D Gaussian process.
    """

    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """
        Class constructor.

        Parameters
        ----------
        f : callable
            Black-box function to optimize.
        X_init : np.ndarray of shape (t, 1)
            Initial input samples.
        Y_init : np.ndarray of shape (t, 1)
            Initial output samples.
        bounds : tuple of (min, max)
            Bounds of the search space.
        ac_samples : int
            Number of acquisition sample points.
        l : float
            Length parameter for the kernel.
        sigma_f : float
            Standard deviation for the outputs.
        xsi : float
            Exploration-exploitation factor.
        minimize : bool
            Whether to perform minimization (True) or maximization (False).
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize


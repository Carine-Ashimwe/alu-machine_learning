#!/usr/bin/env python3
import numpy as np
from 2_gp import GaussianProcess

class BayesianOptimization:
    """Performs Bayesian optimization on a noiseless 1D Gaussian process"""

    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1,
                 sigma_f=1, xsi=0.01, minimize=True):
        self.f = f
        self.gp = GaussianProcess(X_init, Y_init, l, sigma_f)
        self.xsi = xsi
        self.minimize = minimize
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)


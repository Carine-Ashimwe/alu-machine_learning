#!/usr/bin/env python3
import numpy as np
from 0_gp import GaussianProcess  # if 0-gp.py in same folder

class GaussianProcess(GaussianProcess):
    """Extends GaussianProcess with prediction"""

    def predict(self, X_s):
        """Predicts mean and variance of points in Gaussian process"""
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(self.K)

        mu = K_s.T.dot(K_inv).dot(self.Y).reshape(-1)
        sigma = np.diag(K_ss - K_s.T.dot(K_inv).dot(K_s))
        return mu, sigma


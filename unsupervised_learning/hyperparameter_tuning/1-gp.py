#!/usr/bin/env python3
"""
Gaussian Process Prediction
"""

import numpy as np
GP0 = __import__('0-gp').GaussianProcess


class GaussianProcess(GP0):
    """Extends GaussianProcess with prediction method"""

    def predict(self, X_s):
        """
        Predicts the mean and variance of points in the Gaussian process.

        Parameters
        ----------
        X_s : np.ndarray of shape (s, 1)
            Sample points to predict.

        Returns
        -------
        mu : np.ndarray of shape (s,)
            Mean predictions for each point.
        sigma : np.ndarray of shape (s,)
            Variance predictions for each point.
        """
        K = self.K
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(K)

        # Predict mean
        mu = K_s.T.dot(K_inv).dot(self.Y).reshape(-1)

        # Predict variance
        sigma = np.diag(K_ss - K_s.T.dot(K_inv).dot(K_s))

        return mu, sigma



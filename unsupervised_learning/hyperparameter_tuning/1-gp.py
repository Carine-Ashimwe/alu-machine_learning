#!/usr/bin/env python3
"""
Gaussian Process Prediction module
"""

import numpy as np


class GaussianProcess:
    """
    Represents a noiseless 1D Gaussian process and
    allows prediction at new sample points.
    """

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Class constructor.

        Parameters
        ----------
        X_init : np.ndarray of shape (t, 1)
            Initial input samples.
        Y_init : np.ndarray of shape (t, 1)
            Initial output samples.
        l : float
            Length scale parameter for the RBF kernel.
        sigma_f : float
            Standard deviation for the output of the black-box function.
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """
        Computes the covariance kernel matrix using RBF.

        Parameters
        ----------
        X1 : np.ndarray of shape (m, 1)
        X2 : np.ndarray of shape (n, 1)

        Returns
        -------
        K : np.ndarray of shape (m, n)
            Covariance matrix.
        """
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + \
                 np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        return self.sigma_f**2 * np.exp(-0.5 / self.l**2 * sqdist)

    def predict(self, X_s):
        """
        Predicts mean and variance of new sample points.

        Parameters
        ----------
        X_s : np.ndarray of shape (s, 1)
            Points to predict.

        Returns
        -------
        mu : np.ndarray of shape (s,)
            Mean predictions.
        sigma : np.ndarray of shape (s,)
            Variance predictions.
        """
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(self.K)

        # Predict mean
        mu = K_s.T.dot(K_inv).dot(self.Y).reshape(-1)
        # Predict variance
        sigma = np.diag(K_ss - K_s.T.dot(K_inv).dot(K_s))

        return mu, sigma


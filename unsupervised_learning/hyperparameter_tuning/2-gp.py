#!/usr/bin/env python3
"""
Gaussian Process Update module
"""

import numpy as np
GP0 = __import__('1-gp').GaussianProcess


class GaussianProcess(GP0):
    """
    Extends GaussianProcess with the ability to update
    the process with new sample points.
    """

    def update(self, X_new, Y_new):
        """
        Updates the Gaussian process with new sample points.

        Parameters
        ----------
        X_new : np.ndarray of shape (1,)
            New input sample.
        Y_new : np.ndarray of shape (1,)
            Output of the black-box function for X_new.
        """
        # Ensure column vector
        X_new = X_new.reshape(-1, 1)
        Y_new = Y_new.reshape(-1, 1)

        # Update X and Y
        self.X = np.vstack((self.X, X_new))
        self.Y = np.vstack((self.Y, Y_new))

        # Update covariance matrix K
        self.K = self.kernel(self.X, self.X)


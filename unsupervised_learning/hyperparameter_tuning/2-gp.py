#!/usr/bin/env python3
import numpy as np
from 1_gp import GaussianProcess  # if 1-gp.py in same folder

class GaussianProcess(GaussianProcess):
    """Extends GaussianProcess with update"""

    def update(self, X_new, Y_new):
        """Updates GP with new sample"""
        self.X = np.vstack((self.X, X_new.reshape(-1, 1)))
        self.Y = np.vstack((self.Y, Y_new.reshape(-1, 1)))
        self.K = self.kernel(self.X, self.X)


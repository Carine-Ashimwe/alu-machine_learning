#!/usr/bin/env python3
import numpy as np
from 4_bayes_opt import BayesianOptimization

class BayesianOptimization(BayesianOptimization):
    """Extends BayesianOptimization with optimization loop"""

    def optimize(self, iterations=100):
        """Optimizes the black-box function"""
        for _ in range(iterations):
            X_next, _ = self.acquisition()
            if X_next in self.gp.X:
                break
            Y_next = self.f(X_next)
            self.gp.update(X_next, Y_next)

        if self.minimize:
            idx = np.argmin(self.gp.Y)
        else:
            idx = np.argmax(self.gp.Y)

        X_opt = self.gp.X[idx].reshape(1,)
        Y_opt = self.gp.Y[idx].reshape(1,)
        return X_opt, Y_opt


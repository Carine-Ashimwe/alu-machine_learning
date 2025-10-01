#!/usr/bin/env python3
import numpy as np
from scipy.stats import norm
from 3_bayes_opt import BayesianOptimization

class BayesianOptimization(BayesianOptimization):
    """Extends BayesianOptimization with acquisition function"""

    def acquisition(self):
        """Calculates next best sample location using Expected Improvement"""
        mu, sigma = self.gp.predict(self.X_s)
        if self.minimize:
            mu_sample_opt = np.min(self.gp.Y)
            imp = mu_sample_opt - mu - self.xsi
        else:
            mu_sample_opt = np.max(self.gp.Y)
            imp = mu - mu_sample_opt - self.xsi

        with np.errstate(divide='warn'):
            Z = imp / sigma
            EI = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            EI[sigma == 0.0] = 0.0

        X_next = self.X_s[np.argmax(EI)]
        return X_next.reshape(1,), EI


import numpy as np


class BlackScholesModel:
    def __init__(self, size, r, rho, sigma, divid, spot):
        self.size = size
        self.r = r
        self.rho = rho
        self.sigma = sigma
        self.divid = divid
        self.spot = spot
        self.cholesky = np.linalg.cholesky(
            np.eye(size) + rho * (np.ones((size, size)) - np.eye(size))
        )


    def asset(self, T, dates):
        """Generates paths for the asset prices based on the Black-Scholes model."""

        time_step = T / dates
        gaussian = np.random.normal(size=(dates+1, self.size))
        path = np.zeros((dates+1, self.size))
        path[0] = self.spot
        expo = np.exp((self.r - self.divid - np.square(self.sigma) / 2) * time_step)
        for i in range(1, dates + 1):
            for d in range(self.size):
                scale_cholesky_gaussian = np.dot(self.cholesky[d, :], gaussian[i])
                computed_share = path[i - 1, d] * expo[d] * np.exp(self.sigma[d] * np.sqrt(time_step) * scale_cholesky_gaussian)
                path[i, d] = computed_share
        return path


    def asset_conditionally(self, T, dates, n, stocks):
        """"Generates paths for the asset prices based on the Black-Scholes model, conditional on the current stock prices at time n."""
        gaussian = np.random.normal(size=(dates+1-n, self.size))
        path = np.zeros((dates + 1 - n, self.size))
        path[0] = stocks
        for i in range(n + 1, dates + 1):
            time_step = (i - n) * T / float(dates)
            expo = np.exp((self.r - self.divid - np.square(self.sigma) / 2) * time_step)
            for d in range(self.size):
                scale_cholesky_gaussian = np.dot(self.cholesky[d, :], gaussian[i - n])
                computed_share = stocks[d] * expo[d] * np.exp(self.sigma[d] * np.sqrt(time_step) * scale_cholesky_gaussian)
                path[i - n, d] = computed_share
        return path



class Max_call_option:
    def __init__(self, T, dates, size, strike):
        self.T = T
        self.dates = dates
        self.size = size
        self.strike = strike

    def payoff(self, pricesAt, n, r):
        price = max(pricesAt)
        payoff = price - self.strike
        return np.exp(-r * n  * self.T / float(self.dates)) * payoff if payoff > 0 else 0


        
"""A.I.4-7 -- Linear regression optimised with gradient descent."""
import numpy as np


class LinearGD:
    """h(x) = theta_0 + theta_1 x, cost J = 1/(2m) sum (h - y)^2."""

    def __init__(self, theta, alpha):
        self.theta = np.asarray(theta, float)                   # theta^(0) = [theta_0, theta_1]
        self.alpha = alpha                                      # learning rate
        self.history = []                                       # (theta^(t), J(theta^(t))) for t = 0, 1, ...

    def predict(self, x):
        return self.theta[0] + self.theta[1] * x                # h_theta(x)

    def cost(self, x, y):
        r = self.predict(x) - y                                 # residuals r_i
        return (r @ r) / (2 * len(x))                           # J(theta) = 1/(2m) sum r_i^2, Eq. eq:A4_cost

    def gradient(self, x, y):
        r = self.predict(x) - y
        return np.array([r.mean(),                              # dJ/dtheta_0
                         (r * x).mean()])                       # dJ/dtheta_1

    def step(self, x, y):
        if not self.history:
            self.history.append((self.theta.copy(), self.cost(x, y)))
        self.theta = self.theta - self.alpha * self.gradient(x, y)   # simultaneous update, Eq. eq:A4_update
        self.history.append((self.theta.copy(), self.cost(x, y)))
        return self

    def run(self, x, y, n_iter, tol=0.0):
        for _ in range(n_iter):
            if np.linalg.norm(self.gradient(x, y)) < tol:       # stop when |grad J| < tol
                break
            self.step(x, y)
        return self

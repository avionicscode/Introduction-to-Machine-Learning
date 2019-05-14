import numpy as np




"""
  =========================================
|| COPY Codes from the previous assignment ||
  =========================================
"""

class SGD:
    # ========================= EDIT HERE =========================
    def __init__(self, gamma, epsilon):
        self.gamma = gamma
        self.epsilon = epsilon

    def update(self, w, grad, lr):
        updated_weight = w - (grad * lr)
        return updated_weight
    # =============================================================

class Momentum:
    # ========================= EDIT HERE =========================
    def __init__(self, gamma, epsilon):
        self.gamma = gamma
        self.epsilon = epsilon
        self.momentum = 0

    def update(self, w, grad, lr):
        self.momentum = (self.gamma * self.momentum) + (grad *lr)
        updated_weight = w - self.momentum
        return updated_weight
    # =============================================================

class RMSProp:
    # ========================= EDIT HERE =========================
    def __init__(self, gamma, epsilon):
        self.gamma = gamma
        self.epsilon = epsilon
        self.G = 0

    def update(self, w, grad, lr):
        self.G = (self.gamma * self.G) + (1-self.gamma) * (grad**2)
        k = (lr * grad)/np.sqrt(self.G + self.epsilon)
        updated_weight = w - k
        return updated_weight
    # =============================================================
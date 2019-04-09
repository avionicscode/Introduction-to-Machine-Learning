import numpy as np

"""
DO NOT EDIT ANY PARTS OTHER THAN "EDIT HERE" !!! 

[Description]
__init__ - Initialize necessary variables for optimizer class
input   : gamma, epsilon
return  : X

update   - Update weight for one minibatch
input   : w - current weight, grad - gradient for w, lr - learning rate
return  : updated weight 
"""

class SGD:
    def __init__(self, gamma, epsilon):
        # ========================= EDIT HERE =========================
        self.gamma = gamma
        self.epsilon = epsilon
        # =============================================================

    def update(self, w, grad, lr):
        updated_weight = None
        # ========================= EDIT HERE =========================
        updated_weight = w - (grad * lr)
        # =============================================================
        return updated_weight

class Momentum:
    def __init__(self, gamma, epsilon):
        # ========================= EDIT HERE =========================
        self.gamma = gamma
        self.epsilon = epsilon
        self.momentum = 0
        # =============================================================

    def update(self, w, grad, lr):
        updated_weight = None
        # ========================= EDIT HERE =========================
        self.momentum = (self.gamma * self.momentum) + (grad *lr)
        updated_weight = w - self.momentum

        # =============================================================
        return updated_weight


class RMSProp:
    # ========================= EDIT HERE =========================
    def __init__(self, gamma, epsilon):
        # ========================= EDIT HERE =========================
        self.gamma = gamma
        self.epsilon = epsilon
        self.G = 0
        # =============================================================

    def update(self, w, grad, lr):
        updated_weight = None
        # ========================= EDIT HERE =========================
        self.G = (self.gamma * self.G) + (1-self.gamma) * (grad**2)
        k = (lr * grad)/np.sqrt(self.G + self.epsilon)
        updated_weight = w - k

        # =============================================================
        return updated_weight
import numpy as np
class Disconnected_Overlap_G():
    def __init__(self, X, V):
        self.X: np = X
        self.V: np = V
        self.disconnected_G = []

    def add(self, G_new):
        self.disconnected_G.append(G_new)


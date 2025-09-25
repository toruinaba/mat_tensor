import numpy as np
import logging
from src.shell.utils import I_sh, Is_sh, IxI_sh
from src.error import NotConvergedError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


i_sh = np.array([1, 1, 0, 0])
IxI_sh = np.outer(i_sh, i_sh)

class Elastic_sh:  
    def __init__(self, E, n):
        self.E = E
        self.n = n

    @property
    def G(self):
        return self.E / (2 * (1 + self.n))

    @property
    def K(self):
        return self.E / (3 * (1 - 2 * self.n))

    @property
    def A(self):
        return 2 * self.G / (self.K + 4 / 3 * self.G)

    @property
    def De(self):
        return 2 * self.G * Is_sh + self.A * (self.K - 2 / 3 * self.G) * IxI_sh

    @property
    def De_inv(self):
        return np.linalg.inv(self.De)

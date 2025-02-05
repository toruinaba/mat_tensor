import numpy as np
from src.material import Elastic, AF_kinematic
from src.core import Calculator3D

from matplotlib import pyplot as plt

class Test_calculator3D:
    elastic = Elastic(205000.0, 0.3)
    sig_y = 235
    Q = 300.0
    b = 300.0
    C = 40000.0
    k = 300.0

    def test_af_kinematic(self):
        af_kin = AF_kinematic(self.elastic, self.sig_y, self.C, self.k)
        calculator = Calculator3D(af_kin, np.array([0.0, 0.0, 0.0, 200.0, 0.0, 0.0]), 250)
        calculator.calculate_steps()

        x = [e[3] for e in calculator.output.eps_p]
        y = [s[3] for s in calculator.output.sig]

        fig = plt.figure()
        plt.plot(x, y)
        plt.show()

"""
chaboche_n = Chaboche_n(elastic, sig_y, [40000.0, 20000.0, 10000.0, 2000.0, 500.0], [1000.0, 300.0, 100.0, 50.0, 20.0], [200, 100.0], [300.0, 30.0])
calculator = Calculator3D(chaboche_n, np.array([0.0, 0.0, 0.0, 200.0, 0.0, 0.0]), 250)
calculator.calculate_steps()
calculator.goal_sig = np.array([0.0, 0.0, 0.0, -225.0, 0.0, 0.0])
calculator.calculate_steps(is_init=False)
calculator.goal_sig = np.array([0.0, 0.0, 0.0, 250.0, 0.0, 0.0])
calculator.calculate_steps(is_init=False)
calculator.goal_sig = np.array([0.0, 0.0, 0.0, -275.0, 0.0, 0.0])
calculator.calculate_steps(is_init=False)
calculator.goal_sig = np.array([0.0, 0.0, 0.0, 300.0, 0.0, 0.0])
calculator.calculate_steps(is_init=False)



"""
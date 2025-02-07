import numpy as np
from src.material import Elastic
from src.material import (
    Linear_isotropic,
    Linear_kinematic,
    Voce_isotropic,
    Voce_isotropic_n,
    AF_kinematic
)
from src.core import Calculator3D

from matplotlib import pyplot as plt

class Test_calculator3D:
    elastic = Elastic(205000.0, 0.3)
    sig_y = 200
    Q = 300.0
    b = 700.0
    Qs = [200.0, 100.0, 100.0]
    bs = [1000.0, 50.0, 200.0]
    C = 40000.0
    k = 200.0

    def test_linear_isotropic(self):
        linear_iso = Linear_isotropic(self.elastic, self.sig_y, self.C)
        calculator = Calculator3D(linear_iso, np.array([250.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 100)
        calculator.calculate_steps()
        calculator.goal_sig = np.array([-300.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        calculator.calculate_steps(is_init=False)

    def test_linear_kinematic(self):
        linear_kin = Linear_kinematic(self.elastic, self.sig_y, self.C)
        calculator = Calculator3D(linear_kin, np.array([250.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 100)
        calculator.calculate_steps()
        calculator.goal_sig = np.array([-250.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        calculator.calculate_steps(is_init=False)
        calculator.goal_sig = np.array([250.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        calculator.calculate_steps(is_init=False)

    def test_voce_isotropic(self):
        voce_iso = Voce_isotropic(self.elastic, self.sig_y, self.Q, self.b)
        calculator = Calculator3D(voce_iso, np.array([250.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 100)
        calculator.calculate_steps()
        calculator.goal_sig = np.array([-275.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        calculator.calculate_steps(is_init=False)
        calculator.goal_sig = np.array([300.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        calculator.calculate_steps(is_init=False)

    def test_voce_isotropic_n(self):
        voce_iso = Voce_isotropic_n(self.elastic, self.sig_y, self.Qs, self.bs)
        calculator = Calculator3D(voce_iso, np.array([250.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 100)
        calculator.calculate_steps()
        calculator.goal_sig = np.array([-275.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        calculator.calculate_steps(is_init=False)
        calculator.goal_sig = np.array([300.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        calculator.calculate_steps(is_init=False)

    def test_af_kinematic(self):
        af_kin = AF_kinematic(self.elastic, self.sig_y, self.C, self.k)
        calculator = Calculator3D(af_kin, np.array([392.457, 0.0, 0.0, 0.0, 0.0, 0.0]), 1000)
        calculator.calculate_steps()
        calculator.goal_sig = np.array([-399.48, 0.0, 0.0, 0.0, 0.0, 0.0])
        calculator.calculate_steps(is_init=False)
        calculator.goal_sig = np.array([399.9, 0.0, 0.0, 0.0, 0.0, 0.0])
        calculator.calculate_steps(is_init=False)

        x = [e[0] for e in calculator.output.eps_p]
        y = [s[0] for s in calculator.output.sig]

        fig = plt.figure()
        plt.plot(x, y)
        plt.show()

        lines = [f"{x},{y}" for x, y in zip(x, y)]
        with open("./output.csv", "w") as f:
            f.write("\n".join(lines))


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
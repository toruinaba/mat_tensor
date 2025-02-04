import numpy as np
from material import Elastic, Chaboche_n
from core import Calculator3D

E = 205000.0
n = 0.3
sig_y = 235
Q = 300.0
b = 300.0

elastic = Elastic(E, n)
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


x = [e[3] for e in calculator.output.eps_p]
y = [s[3] for s in calculator.output.sig]

from matplotlib import pyplot as plt

fig = plt.figure()
plt.plot(x, y)
plt.show()

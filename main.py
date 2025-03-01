import numpy as np
from material import Elastic, AF_kinematic, Chaboche, Yoshida_uemori
from src.core import Calculator3D

E = 206000.0
n = 0.3
sig_y = 200

elastic = Elastic(E, n)
chaboche = Chaboche(elastic, 200.0, 20000.0, 30.0, 300.0, 100.0)
yu = Yoshida_uemori(elastic, 124.0, 168.0, 500.0, 190.0, 12.0, 9.0, 0.1)
calculator = Calculator3D(yu, np.array([202.5, 0.0, 0.0, 0.0, 0.0, 0.0]), 50)
calculator.calculate_steps()
calculator.goal_sig = np.array([-239.5, 0.0, 0.0, 0.0, 0.0, 0.0])
calculator.calculate_steps(is_init=False)
calculator.goal_sig = np.array([243.65, 0.0, 0.0, 0.0, 0.0, 0.0])
calculator.calculate_steps(is_init=False)
calculator.goal_sig = np.array([-243.85, 0.0, 0.0, 0.0, 0.0, 0.0])
calculator.calculate_steps(is_init=False)
calculator.goal_sig = np.array([244, 0.0, 0.0, 0.0, 0.0, 0.0])
calculator.calculate_steps(is_init=False)

x = [e[0] for e in calculator.output.eps_p]
y = [s[0] for s in calculator.output.sig]

from matplotlib import pyplot as plt

fig = plt.figure()
plt.plot(x, y)
plt.show()

x = [e for e in calculator.output.eff_eps_p]
y = [r for r in calculator.output.R]

fig = plt.figure()
plt.plot(x, y)
plt.show()

x = [e for e in calculator.output.eff_eps_p]
y = [r for r in calculator.output.r]
y2 = [gr for gr in calculator.output.gr]

fig = plt.figure()
plt.plot(x, y)
plt.plot(x, y2)
plt.show()
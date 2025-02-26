import numpy as np
from material import Elastic, AF_kinematic, Chaboche, Yoshida_uemori
from src.core import Calculator3D

E = 205000.0
n = 0.3
sig_y = 200

elastic = Elastic(E, n)
chaboche = Chaboche(elastic, 200.0, 20000.0, 30.0, 300.0, 100.0)
yu = Yoshida_uemori(elastic, 124.0, 168.0, 500.0, 190.0, 12.0, 9.0, 0.5)
calculator = Calculator3D(yu, np.array([0.0, 0.0, 0.0, 100.0, 0.0, 0.0]), 20)
calculator.calculate_steps()
calculator.goal_sig = np.array([0.0, 0.0, 0.0, -120.0, 0.0, 0.0])
calculator.calculate_steps(is_init=False)
calculator.goal_sig = np.array([0.0, 0.0, 0.0, 140.0, 0.0, 0.0])
calculator.calculate_steps(is_init=False)
calculator.goal_sig = np.array([0.0, 0.0, 0.0, -160.0, 0.0, 0.0])
calculator.calculate_steps(is_init=False)

x = [e[3] for e in calculator.output.eps_p]
y = [s[3] for s in calculator.output.sig]

from matplotlib import pyplot as plt

fig = plt.figure()
plt.plot(x, y)
plt.show()

lines = [f"{x},{y}" for x, y in zip(x, y)]
with open("./output.csv", "w") as f:
    f.write("\n".join(lines))
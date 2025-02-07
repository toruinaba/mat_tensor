import numpy as np
from material import Elastic, AF_kinematic, Chaboche
from src.core import Calculator3D

E = 205000.0
n = 0.3
sig_y = 200

elastic = Elastic(E, n)
chaboche = Chaboche(elastic, 200.0, 20000.0, 30.0, 300.0, 100.0)
calculator = Calculator3D(chaboche, np.array([0.0, 0.0, 312.581, 0.0, 0.0, 0.0]), 300)
calculator.calculate_steps()
calculator = Calculator3D(chaboche, np.array([0.0, 0.0, -393.986, 0.0, 0.0, 0.0]), 300)
calculator.calculate_steps()
calculator = Calculator3D(chaboche, np.array([0.0, 0.0, 434.79, 0.0, 0.0, 0.0]), 300)
calculator.calculate_steps()


x = [e[2] for e in calculator.output.eps_p]
y = [s[2] for s in calculator.output.sig]

from matplotlib import pyplot as plt

fig = plt.figure()
plt.plot(x, y)
plt.show()

lines = [f"{x},{y}" for x, y in zip(x, y)]
with open("./output.csv", "w") as f:
    f.write("\n".join(lines))
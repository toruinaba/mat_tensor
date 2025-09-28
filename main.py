import numpy as np
from src.solid.materials import Elastic, AF_kinematic, Chaboche, Yoshida_uemori
from src.solid.calculator import Calculator3D

E = 206000.0
n = 0.3
sig_y = 200

elastic = Elastic(E, n)
chaboche = Chaboche(elastic, 200.0, 20000.0, 30.0, 300.0, 100.0)
yu = Yoshida_uemori(elastic, 124.0, 168.0, 500.0, 190.0, 12.5, 9.3, 0.5, 159000.0, 0.0)

a1 = [202.5, -244, 249., -248.87, 250.0]
a2 = [x / 2 for x in a1]
a3 = [x / np.sqrt(3) / 1.6 for x in a1]

idx = 2  # 0: exx, 1: eyy, 2: ezz, 3: gxy, 4: gyz, 5: gzx

goal_sig = np.zeros(6)
goal_sig[idx] = a1[0]
goal_sig[idx+1] = a2[0]
goal_sig[idx+2] = a3[0]

calculator = Calculator3D(yu, goal_sig, 0.01, 1.0e-05, 0.01)
calculator.calculate_steps()

if len(a1) >= 2:
    for iamp in range(1, len(a1)):
        calculator.goal_sig[idx] = a1[iamp]
        calculator.goal_sig[idx+1] = a2[iamp]
        calculator.goal_sig[idx+2] = a3[iamp]
        calculator.calculate_steps(is_init=False)

x = [e[idx] for e in calculator.output.eps]
y = [s[idx] for s in calculator.output.sig]

from matplotlib import pyplot as plt

fig = plt.figure()
plt.plot(x, y)
plt.show()

import numpy as np
from src.material import Elastic, AF_kinematic, Chaboche, Yoshida_uemori
from src.core import Calculator3D

E = 206000.0
n = 0.3
sig_y = 200

elastic = Elastic(E, n)
chaboche = Chaboche(elastic, 200.0, 20000.0, 30.0, 300.0, 100.0)
yu = Yoshida_uemori(elastic, 124.0, 168.0, 500.0, 190.0, 12.0, 9.0, 0.5, 159000.0, 30.8)

a = [202.5, -244, 249.2, -248.87, 249.3]
amps = [x for x in a]

idx = 0

goal_sig = np.zeros(6)
goal_sig[idx] = amps[0]

calculator = Calculator3D(yu, goal_sig)
calculator.calculate_steps()

if len(amps) >= 2:
    for iamp in amps[1:]:
        calculator.goal_sig[idx] = iamp
        calculator.calculate_steps(is_init=False)

x = [e[idx] for e in calculator.output.eps]
y = [s[idx] for s in calculator.output.sig]

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
y2 = [g for g in zip(calculator.output.gr)]
fig = plt.figure()
plt.plot(x, y)
plt.plot(x, y2)
plt.show()
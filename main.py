import numpy as np
from src.material import Elastic,AF_kinematic, Chaboche, Yoshida_uemori
from src.core import Calculator3D, Calculator_shell
from src.material_shell import Linear_isotropic_shell, Elastic_shell

E = 206000.0
n = 0.3
sig_y = 200
h = 20000.0

elastic = Elastic_shell(E, n)

"""
chaboche = Chaboche(elastic, 200.0, 20000.0, 30.0, 300.0, 100.0)
yu = Yoshida_uemori(elastic, 124.0, 168.0, 500.0, 190.0, 12.5, 9.3, 0.5, 159000.0, 30.8)
"""

linear_iso = Linear_isotropic_shell(elastic, sig_y, h)

a = [250, -300, 350]#[202.5, -244, 249.2, -248.87, 249.3]
amps = [x for x in a]

idx = 0

goal_sig = np.zeros(3)
goal_sig[idx] = amps[0]

calculator = Calculator_shell(linear_iso, goal_sig, 0.01, 1.0e-05, 0.01)
calculator.calculate_steps()

if len(amps) >= 2:
    for iamp in amps[1:]:
        calculator.goal_sig[idx] = iamp
        calculator.calculate_steps(is_init=False)

x = [e[idx] for e in calculator.output.eps_p]
y = [s[idx] for s in calculator.output.sig]

from matplotlib import pyplot as plt

fig = plt.figure()
plt.plot(x, y)
plt.plot([-0.01, 0.01], [200.0, 200.0])
plt.show()

for i in range(len(calculator.output.eff_eps_p)):
    grad = (calculator.output.mises[i] - calculator.output.mises[i-1]) / (calculator.output.eff_eps_p[i] - calculator.output.eff_eps_p[i-1])
    print(f"grad: {grad}")

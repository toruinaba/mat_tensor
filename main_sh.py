from src.shell.materials import (
    Elastic_sh,
    Linear_isotropic_sh,
    Linear_kinematic_sh,
    Voce_isotropic_sh,
    AF_kinematic_sh,
    Yoshida_uemori_sh,
    Yoshida_uemori_sh2,
)
from src.solid.materials import (
    Elastic,
    Linear_isotropic,
    Linear_kinematic,
    Voce_isotropic,
    AF_kinematic,
    Yoshida_uemori,
)
from src.solid.calculator import Calculator3D
from src.shell.calculator import Calculator3D_sh
import numpy as np

E = 205000.0
n = 0.3
elastic = Elastic(E, n)
elastic_sh = Elastic_sh(E, n)
"""
# linear isotropic
lin_iso = Linear_isotropic(elastic, 200.0, 20000.0)
lin_iso_sh = Linear_isotropic_sh(elastic_sh, 200.0, 20000.0)

# linear kinematic
lin_kin = Linear_kinematic(elastic, 200.0, 20000.0)
lin_kin_sh = Linear_kinematic_sh(elastic_sh, 200.0, 20000.0)

# Voce isotropic
voce_iso = Voce_isotropic(elastic, 200.0, 10000.0, 10.0)
voce_iso_sh = Voce_isotropic_sh(elastic_sh, 200.0, 10000.0, 10.0)

# Armstrong-Frederick kinematic
af_kin = AF_kinematic(elastic, 200.0, 10000.0, 25.0)
af_kin_sh = AF_kinematic_sh(elastic_sh, 200.0, 10000.0, 25.0)
"""

# Yoshida-Uemori
yu = Yoshida_uemori(elastic, 124.0, 168.0, 500.0, 190.0, 12.5, 9.3, 0.5, 159000.0, 0.3)
yu_sh = Yoshida_uemori_sh(
    elastic_sh, 124.0, 168.0, 500.0, 190.0, 12.5, 9.3, 0.5, 159000.0, 0.3
)
yu_sh2 = Yoshida_uemori_sh2(
    elastic_sh, 124.0, 168.0, 500.0, 190.0, 12.5, 9.3, 0.5, 159000.0, 0.3
)

a = [202.5, -220, 249.0, -248.87, 250.0]
# a2 = [x / 2 for x in a1]
a1 = [x / np.sqrt(3) / 1.5 for x in a]

idx = 3  # 0: exx, 1: eyy, 2: ezz, 3: gxy, 4: gyz, 5: gzx
idx_sh = 2  # 0: exx, 1: eyy, 2: gxy

goal_sig = np.zeros(6)
goal_sig[idx] = a1[0]

goal_sig_sh = np.zeros(3)
goal_sig_sh[idx_sh] = a1[0]
# goal_sig[idx+1] = a2[0]
# goal_sig[idx+2] = a3[0]

calculator = Calculator3D(yu, goal_sig, 0.01, 1.0e-05, 0.01)
calculator.calculate_steps()

if len(a1) >= 2:
    for iamp in range(1, len(a1)):
        calculator.goal_sig[idx] = a1[iamp]
        calculator.calculate_steps(is_init=False)

calculator_sh = Calculator3D_sh(yu_sh2, goal_sig_sh, 0.01, 1.0e-05, 0.01)
calculator_sh.calculate_steps()

if len(a1) >= 2:
    for iamp in range(1, len(a1)):
        calculator_sh.goal_sig[idx_sh] = a1[iamp]
        calculator_sh.calculate_steps(is_init=False)


x = [e[idx] for e in calculator.output.eps]
y = [s[idx] for s in calculator.output.sig]

x2 = [e[idx_sh] for e in calculator_sh.output.eps]
y2 = [s[idx_sh] for s in calculator_sh.output.sig]

from matplotlib import pyplot as plt

fig = plt.figure()
plt.plot(x, y)
plt.plot(x2, y2, linestyle="dashed")
plt.show()

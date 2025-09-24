import numpy as np
from src.axisymmetry.materials import (
    Elastic_ax,
    Linear_isotropic_ax,
    Linear_kinematic_ax,
    Voce_isotropic_ax,
    AF_kinematic_ax,
    Yoshida_uemori_ax
)
from src.core import Calculator3D_ax

E = 206000.0
n = 0.3
sig_y = 200

elastic = Elastic_ax(E, n)
lin_iso = Linear_isotropic_ax(elastic, sig_y, 2000.0)
lin_kin = Linear_kinematic_ax(elastic, sig_y, 2000.0)
voce_iso = Voce_isotropic_ax(elastic, sig_y, 2000.0, 5.0)
af_kin = AF_kinematic_ax(elastic, sig_y, 2000.0, 20.0)
yu = Yoshida_uemori_ax(elastic, 124.0, 168.0, 500.0, 190.0, 12.5, 9.3, 0.5, 159000.0, 0.0)

a1 = [205, -244, 249.5, -255.0]
#a2 = [x / 2 for x in a1]
#a3 = [x / np.sqrt(3) / 1.6 for x in a1]

idx = 0  # 0: exx, 1: eyy, 2: ezz, 3: gxy, 4: gyz, 5: gzx

goal_sig = np.zeros(4)
goal_sig[idx] = a1[0]
#goal_sig[idx+1] = a2[0]
#goal_sig[idx+2] = a3[0]

calculator = Calculator3D_ax(yu, goal_sig, 0.01, 1.0e-05, 0.01)
calculator.calculate_steps()

if len(a1) >= 2:
    for iamp in range(1, len(a1)):
        calculator.goal_sig[idx] = a1[iamp]
        #calculator.goal_sig[idx+1] = a2[iamp]
        #calculator.goal_sig[idx+2] = a3[iamp]
        calculator.calculate_steps(is_init=False)

x = [e[idx] for e in calculator.output.eps]
y = [s[idx] for s in calculator.output.sig]

from matplotlib import pyplot as plt

fig = plt.figure()
plt.plot(x, y)
plt.show()


import numpy as np

I_sh = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

Is_sh = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0.5]])

i_sh = np.array([1, 1, 0])

IxI_sh = np.outer(i_sh, i_sh)

print(IxI_sh)
Id_sh = Is_sh - 1 / 3 * IxI_sh
Id_s_sh = I_sh - 1 / 3 * IxI_sh


P = 1 / 3 * np.array([[2, -1, 0], [-1, 2, 0], [0, 0, 6]])

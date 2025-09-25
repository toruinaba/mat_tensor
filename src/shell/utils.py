import numpy as np

I_sh = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 0]
])

Is_sh = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0.5, 0],
    [0, 0, 0, 0]
])

i_sh = np.array([1, 1, 0, 0])

IxI_sh = np.outer(i_sh, i_sh)

Id_sh = Is_sh - 1 / 3 * IxI_sh
Id_s_sh = I_sh - 1 / 3 * IxI_sh
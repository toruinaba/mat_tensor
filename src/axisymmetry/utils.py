import numpy as np

I_ax = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

Is_ax = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0.5, 0],
    [0, 0, 0, 1]
])

i_ax = np.array([1, 1, 0, 1])

IxI_ax = np.outer(i_ax, i_ax)

Id_ax = Is_ax - 1 / 3 * IxI_ax
Id_s_ax = I_ax - 1 / 3 * IxI_ax
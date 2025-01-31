import numpy as np


Ni = np.array([
    [1, 1, 1, 1, 1, 1],
    [2, 2, 2, 2, 2, 2],
    [3, 3, 3, 3, 3, 3],
    [1, 1, 1, 1, 1, 1],
    [2, 2, 2, 2, 2, 2],
    [1, 1, 1, 1, 1, 1]
])

Nj = np.array([
    [1, 1, 1, 1, 1, 1],
    [2, 2, 2, 2, 2, 2],
    [3, 3, 3, 3, 3, 3],
    [2, 2, 2, 2, 2, 2],
    [3, 3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3, 3]
])

Nk = np.array([
    [1, 2, 3, 1, 2, 1],
    [1, 2, 3, 1, 2, 1],
    [1, 2, 3, 1, 2, 1],
    [1, 2, 3, 1, 2, 1],
    [1, 2, 3, 1, 2, 1],
    [1, 2, 3, 1, 2, 1]
])

Nl = np.array([
    [1, 2, 3, 2, 3, 3],
    [1, 2, 3, 2, 3, 3],
    [1, 2, 3, 2, 3, 3],
    [1, 2, 3, 2, 3, 3],
    [1, 2, 3, 2, 3, 3],
    [1, 2, 3, 2, 3, 3]
])

def kron_delta(A, B):
    if A.shape != B.shape:
        raise ValueError("Different size matrix is provided.")
    (m, n) = np.shape(A)
    mat = np.zeros(np.shape(A))
    for i_m in range(m):
        for i_n in range(n):
            if A[i_m, i_n] == B[i_m, i_n]:
                mat[i_m, i_n] = 1
    return mat

I = kron_delta(Ni, Nk) * kron_delta(Nj, Nl)
I_t = kron_delta(Ni, Nl) * kron_delta(Nj, Nk)
IxI = kron_delta(Ni, Nj) * kron_delta(Nk, Nl)
Is = 0.5 * (I + I_t)
i = np.array([1, 1, 1, 0, 0, 0])

Id = Is - 1 / 3 * IxI
from re import S
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

E = 205000.0
n = 0.0
sig_y = 235
H = 200.0
b = 30.0

G = E / (2 * (1 + n))
A = 1
K = E /(3 * (1 - 2 * n))

De = 2 * G * Is + 1 * (K - 2 / 3 * G) * IxI

sig = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
eps = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
eps_e = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
eps_p = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
eff_eps_p = 0.0

STEP = 1000

def calc_yield_stress(eff_eps_p):
    return sig_y + H * (1 - np.exp(-b * eff_eps_p))


def calc_plastic_mod(eff_eps_p):
    return H * b * np.exp(-b * eff_eps_p)

miseses = []
eff_eps_ps = []

for inc in range(STEP):
    print(f"ITERATION {inc}")
    del_eps_tri = np.array([0.00001, 0, 0, 0, 0, 0])
    eps += del_eps_tri
    eps_e_tri = eps_e + del_eps_tri
    eps_v_tri = 1 / 3 * (IxI) @ eps_e_tri
    eps_v_tri_tr = (eps_v_tri[0] + eps_v_tri[1] + eps_v_tri[2])
    p_tri = K * eps_v_tri_tr

    eps_d_tri = Id @ eps

    sig_d_tri = 2 * G * eps_d_tri

    q_tri = np.sqrt(3 / 2 * sig_d_tri @ sig_d_tri)
    y = calc_yield_stress(eff_eps_p)

    f_tri = q_tri - y

    if f_tri >= 0:
        # yield
        del_gam = 0.0
        for itr in range(10):
            h = calc_plastic_mod(eff_eps_p + del_gam)
            d = -3 * G - h
            diff = q_tri - 3 * G * del_gam - calc_yield_stress(eff_eps_p + del_gam)
            print(diff)
            if abs(diff) < 0.01:
                print("Converged")
                break
            if itr == 9:
                raise ValueError("NotConverged")
            del_gam -= f_tri / d
            
        p = p_tri
        sig_d = (1 - del_gam * 3 * G / q_tri) * sig_d_tri
        sig = sig_d + p_tri * i
        eps_e = 1 / (2 * G) * sig_d + 1 / 3 * eps_v_tri_tr * I
        eff_eps_p += del_gam
        miseses.append(np.sqrt(3 / 2 * sig_d @ sig_d))
        eff_eps_ps.append(eff_eps_p)

    else:
        sig = sig_d_tri + p_tri * i
        eps_e = eps_e_tri
        eps_p = eps_p
        miseses.append(q_tri)
        eff_eps_ps.append(eff_eps_p)

from matplotlib import pyplot as plt

fig = plt.figure()
plt.plot(eff_eps_ps, miseses)
plt.show()

print(eff_eps_ps)


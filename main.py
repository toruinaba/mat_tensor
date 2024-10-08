import numpy as np

TOL = 1.0e-06
STEP = 100

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
n = 0.3
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

goal_sig = np.array([400.0, 0.0, 0.0, 0.0, 0.0, 0.0])

def calc_yield_stress(eff_eps_p):
    return sig_y + H * (1 - np.exp(-b * eff_eps_p))


def calc_plastic_mod(eff_eps_p):
    return H * b * np.exp(-b * eff_eps_p)

miseses = []
eff_eps_ps = []

for inc in range(STEP):
    print(f"ITERATION {inc}")
    del_eps_tri = np.array([0.0001, -0.0001*n, -0.0001*n, 0, 0, 0])
    eps += del_eps_tri
    eps_e_tri = eps - eps_p
    eps_e_d_tri = Id @ eps_e_tri
    eps_e_v_tri = 1 / 3 * IxI @ eps_e_tri
    eps_e_v_tri_tr = eps_e_v_tri[0] + eps_e_v_tri[1] + eps_e_v_tri[2]
    p_tri = K * eps_e_v_tri_tr

    sig_d_tri = 2 * G * eps_e_d_tri
    q_tri = np.sqrt(3 / 2 * sig_d_tri @ sig_d_tri)
    n_vector = sig_d_tri / np.sqrt(sig_d_tri @ sig_d_tri)
    
    y = calc_yield_stress(eff_eps_p)
    phi = q_tri - y
    hd = calc_plastic_mod(eff_eps_p)
    
    del_gam = 0.0
    if phi >= 0:
        for itr in range(10):
            diff = q_tri - 3 * G * del_gam - y
            del_gam += diff / (3 * G + hd)
            y = calc_yield_stress(eff_eps_p + del_gam)
            if abs(diff) < TOL * y:
                print("Converged")
                break
    eps_e_d = eps_e_d_tri - 3 / 2 * del_gam * n_vector
    eps_p += 3 / 2 * del_gam * n_vector
    eps_e = eps - eps_p
    eff_eps_p += del_gam
    sig_d = 2 * G * eps_e_d
    mises = np.sqrt(3 / 2 * sig_d @ sig_d)
    sig_v = p_tri * i
    sig = sig_d + sig_v

    
    miseses.append(mises)
    eff_eps_ps.append(eff_eps_p)

    if inc >= 13:
        break

from matplotlib import pyplot as plt

fig = plt.figure()
plt.plot(eff_eps_ps, miseses)
plt.show()

print(eff_eps_ps)


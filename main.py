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

TOL = 1.0e-06
STEP = 600
RM_I = 10
NW_I = 100

E = 205000.0
n = 0.3
sig_y = 235
Q = 300.0
b = 150.0

G = E / (2 * (1 + n))
A = 1
K = E /(3 * (1 - 2 * n))

De = 2 * G * Is + 1 * (K - 2 / 3 * G) * IxI

sig = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
eps = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
eps_e = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
eps_p = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
eff_eps_p = 0.0
r = 0.0

goal_sig = np.array([800.0, 0.0, 0.0, 0.0, 0.0, 0.0])


sig_hist = []
eps_hist = []
eps_e_hist = []
eps_p_hist = []
eff_eps_p_hist = []
r_hist = []
mises_hist = []

for i in range(STEP):
    print("="*80)
    print(f"Increment {i}")
    del_eps = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    sig_ip1 = sig + np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    Dep = De
    f_sig = sig_ip1 - sig
    del_eps = np.linalg.inv(Dep) @ f_sig
    print(f"Initial dela eps: {del_eps}")

    for itr in range(NW_I):
        print("-"*80)
        print(f"Iteration: {itr+1}")
        eps_tri = eps + del_eps
        eps_e_tri = eps_tri - eps_p
        eps_e_d_tri = Id @ eps_e_tri
        sig_tri = De @ eps_e_tri
        sig_d_tri = Id @ sig_tri
        sig_v_tri = 1 / 3 * IxI @ sig_tri

        q_tri = np.sqrt(3/2 * sig_d_tri @ sig_d_tri)
        n_bar = sig_d_tri / q_tri
        del_gam = 0.0
        theta = 1.0

        f_ip1 = sig_y + r - q_tri # applied del_gam = 0, theta = 0
        f_ip1_prime = 3 * G - b * (r + Q) # applied del_gam = 0, theta = 0
        if f_ip1 < 0.0:
            print("Plastic behavior")
            for inew in range(RM_I):
                print(f"Return map iteration {inew+1}")
                print(f_ip1)
                d_del_gam = f_ip1 / f_ip1_prime
                del_gam -= d_del_gam
                theta = 1 / (1 + b * del_gam)
                f_ip1 = sig_y + theta * (r + b * Q * del_gam) + 3 * G * del_gam - q_tri
                f_ip1_prime = 3 * G - b * theta**2 * (r + b * Q * del_gam) + theta * b * Q
                if abs(f_ip1) < TOL:
                    if del_gam < 0.0:
                        raise ValueError("Delta gamma is negative value.")
                    print(f"Return map converged itr.{inew+1}")
                    print(f"Delta gamma: {del_gam}")
                    break
                if inew == RM_I - 1:
                    raise ValueError("Return map isn't converged")
            delta_eps_p = np.sqrt(3 / 2) * del_gam * n_bar
            eps_p_ip1 = eps_p + delta_eps_p
            eps_e_ip1 = eps_tri - eps_p_ip1
            eps_e_d_ip1 = Id @ eps_e_ip1
            eps_ip1 = eps_tri
            r_ip1 = theta * (r + b * Q * del_gam)
            eff_eps_p_ip1 = eff_eps_p + del_gam
            sig_r = 2 * G * eps_e_d_ip1 + sig_v_tri
        else:
            print("Elastic behavior")
            eps_ip1 = eps_tri
            eps_e_ip1 = eps_e_tri
            eps_p_ip1 = eps_tri - eps_e_tri
            r_ip1 = r
            eff_eps_p_ip1 = eff_eps_p
            sig_r = De @ eps_e_ip1

        sig_diff = sig_ip1 - sig_r
        print(f"Corrected sig: {sig_r}")
        print(f"Difference norm: {np.sqrt(sig_diff @ sig_diff)}")
        if np.sqrt(3 / 2 * sig_diff @ sig_diff) / (sig_y + r) < TOL:
            sig = sig_ip1
            sig_hist.append(sig)
            eps_e = eps_e_ip1
            eps_e_hist.append(eps_e)
            eps_p = eps_p_ip1
            eps_p_hist.append(eps_p)
            eps = eps_ip1
            eps_hist.append(eps)
            r = r_ip1
            r_hist.append(r)
            eff_eps_p = eff_eps_p_ip1
            eff_eps_p_hist.append(eff_eps_p)
            sig_d = Id @ sig
            mises = np.sqrt(3/2 * sig_d @ sig_d)
            mises_hist.append(mises)
            print(f"Iteration converged: itr.{itr+1}")
            print(f"Mises: {mises}")
            break
        else:
            print(f"itr.{itr+1}")
            Dep = De - 6 * G**2 * del_gam / q_tri * Id + 6 * G**2 * (del_gam / q_tri - 1 / f_ip1_prime) * np.outer(n_bar, n_bar)
            d_del_eps = np.linalg.inv(Dep) @ sig_diff
            del_eps += d_del_eps
            print(f"Updated delta eps: {del_eps}")
        if itr == NW_I - 1:
            raise ValueError(f"Not converged this iteration.")

from matplotlib import pyplot as plt

fig = plt.figure()
e11 = [x[0] for x in eps_p_hist]
s11 = [x[0] for x in sig_hist]
plt.plot(e11, s11)
plt.show()

import numpy as np

TOL = 1.0e-06
STEP = 125

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
H = 20000
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
    return sig_y + H *eff_eps_p


def calc_plastic_mod(eff_eps_p):
    return H

def return_mapping(eps_e_d_tri, eff_eps_p):
    sig_d_tri = 2 * G * eps_e_d_tri
    q_tri = np.sqrt(3 / 2 * sig_d_tri @ sig_d_tri)
    if q_tri == 0.0:
        n_vector = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    else:
        n_vector = sig_d_tri / np.sqrt(sig_d_tri @ sig_d_tri)
    
    y = calc_yield_stress(eff_eps_p)
    phi = q_tri - y
    hd = calc_plastic_mod(eff_eps_p)
    print(f"Yield stress: {y}")
    print(f"Phi: {phi}")
    
    del_gam = 0.0
    if phi >= 0.0001:
        for itr in range(10):
            diff = q_tri - 3 * G * del_gam - y
            print(f"Return map difference: {diff}")
            if abs(diff) < TOL * y:
                print("Return map converged")
                break
            del_gam += diff / (3 * G + hd)
            y = calc_yield_stress(eff_eps_p + del_gam)
    print(f"Q_tri: {q_tri}")
    eps_e_d = (1 -  3 * del_gam * G / q_tri) * eps_e_d_tri
    sig_d = 2 * G * eps_e_d
    print(f"Corrected sig_d: {sig_d}")
    mises = np.sqrt(3 / 2 * sig_d @ sig_d)
    print(f"Corrected Mises: {mises}")
    print(f"Current Y: {calc_yield_stress(eff_eps_p + del_gam)}")
    return q_tri, del_gam, n_vector, hd

def calc_Dep(eps, eps_p, eff_eps_p):
    eps_e_tri = eps - eps_p

    eps_e_d_tri = Id @ eps_e_tri
    q_tri, del_gam, n_vector, hd = return_mapping(eps_e_d_tri, eff_eps_p)
    NxN = np.matrix(n_vector).transpose() @ np.matrix(n_vector)

    if q_tri == 0.0:
        Dep = 2 * G * Id + K * IxI
    else:
        sig_e_d = 2 * G * (1 - 3 * del_gam * G / q_tri) * eps_e_d_tri
        sig_e_v = K * IxI @ eps
        print(f"Sig_e_d: {sig_e_d}")
        print(f"Sig_e_v: {sig_e_v}")
        print(f"Mises: {np.sqrt(3 / 2 * sig_e_d @ sig_e_d)}")
        Dep = (
            2 * G * (1 - 3 * del_gam * G / q_tri) * Id +
            6 * G**2 * (del_gam / q_tri - 1 / (3 * G + hd)) * NxN * (1.0 if del_gam != 0.0 else 0.0) +
            K * IxI
        )
    return np.array(Dep), del_gam, n_vector, q_tri

miseses = []
eff_eps_ps = []


pl_flag = False

for inc in range(STEP):
    print("-"*30 + f"INCREMENT {inc}" + "-"*30)
    del_sig = np.array([2.0, 0, 0, 0, 0, 0])
    sig_tri = sig + del_sig
    print(f"Goal Sigma: {sig_tri}")
    print(f"Current Eps: {eps}")
    del_eps = np.array([0.0]*6)
    d_del_eps = np.array([0.0]*6)
    eps_tri = eps + del_eps
    del_sig_ep = np.array([0.0]*6)
    del_eps_p = np.array([0.0]*6)
    del_eff_eps_p = 0.0

    Dep, del_gam, n_vector, q_tri = calc_Dep(eps, eps_p, eff_eps_p)
    del_sig_ep = Dep @ d_del_eps

    for itr in range(10):
        print(f"Iteration {itr}")
        print(f"Eps_tri on Iter.{itr}: {eps_tri}")
        print(f"Delta gamma: {del_gam}")
        print(f"N vector: {n_vector}")
        print(f"Del sig ep: {del_sig_ep}")
        corrected_sig = sig + del_sig_ep
        sig_diff = sig_tri - corrected_sig
        print(f"Sigma difference: {sig_diff}")
        diff_norm = np.sqrt(sig_diff @ sig_diff)
        if diff_norm  < 0.0001:
            print("ResForce Converged")
            break
        if itr == 9:
            raise ValueError("NotConverged")
        d_del_eps = np.linalg.inv(Dep) @ sig_diff
        del_eps += d_del_eps
        del_eps_p += 3 / 2 * del_gam * n_vector
        del_eff_eps_p += del_gam
        del_sig_ep += Dep @ d_del_eps

        eps_tri = eps + del_eps
        eps_p_tri = eps_p + del_eps_p
        eff_eps_p_tri = eff_eps_p + del_eff_eps_p
        Dep, del_gam, n_vector, q_tri = calc_Dep(eps_tri, eps_p_tri, eff_eps_p_tri)
        print(f"Delta Eps tri on Iter.{itr}: {d_del_eps}")
        print(f"Updated Eps Tri on Iter.{itr}: {eps_tri}")
        print(f"Updated Eps_p Tri on Iter.{itr}: {eps_p_tri}")
        print(f"Updated Eff_eps_p Tri on Iter.{itr}: {eff_eps_p_tri}")

        
    
    eps += del_eps
    sig = sig_tri
    print(f"Updated Eps: {eps}")
    eps_p += del_eps_p
    eps_e = eps - eps_p
    eps_e_d = Id @ eps_e
    sig_d = 2 * G * eps_e_d
    eff_eps_p += del_eff_eps_p 
    print(f"Updated Eps_e: {eps_e}")
    print(f"Updated Eps_p: {eps_p}")
    print(f"Updated eff_eps_p: {eff_eps_p}")
    mises = np.sqrt(3 / 2 * sig_d @ sig_d)
    print(f"Mises: {mises}")
    print(f"Current Sig_y: {calc_yield_stress(eff_eps_p)}")
    if eff_eps_p != 0.0:
        print(f"Hardening mod: {(mises - miseses[-1])/(eff_eps_p - eff_eps_ps[-1])}")
    eff_eps_ps.append(eff_eps_p)
    miseses.append(mises)


from matplotlib import pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(eff_eps_ps, miseses)
ax.plot([0, 0.002], [sig_y, sig_y])
plt.show()



import numpy as np

from util import Is, Id, IxI

TOL = 1.0e-06
STEP = 600
RM_I = 10
NW_I = 100


class Elastic:
    def __init__(self, E: float, n: float):
        self.E = E
        self.n = n

    @property
    def G(self):
        return self.E / (2 * (1 + self.n))

    @property
    def K(self):
        return self.E / (3 * (1 - 2 * self.n))

    @property
    def De(self):
        return 2 * self.G * Is + 1 * (self.K - 2 / 3 * self.G) * IxI

    @property
    def De_inv(self):
        return np.linalg.inv(self.De)


class Plastic_behavior_base:
    def __init__(self, elastic: Elastic, sig_y: float):
        self.elastic = elastic
        self.sig_y = sig_y

    def initialize(self):
        raise NotImplementedError()

    def update(self):
        raise NotImplementedError()

    def calc_f_ip1(self):
        raise NotImplementedError()

    def calc_f_ip1_prime(self):
        raise NotImplementedError()

    def calc_Dep(self):
        raise NotImplementedError()


class Isotropic_voce(Plastic_behavior_base):
    def __init__(self, elastic: Elastic, sig_y: float, Q: float, b: float):
        super().__init__(elastic, sig_y)
        self.Q = Q
        self.b = b

    def initialize(self):
        self.r = 0.0
        self.r_i = 0.0
        self.eps_p = np.array([0.0]*6)
        self.eps_p_i = np.array([0.0]*6)

    def update_i(self, del_gam, n_bar):
        theta = 1 / (1 + self.b * del_gam)
        self.r_i = theta * (self.r + self.b * self.Q * del_gam)
        delta_eps_p = np.sqrt(3 / 2) * del_gam * n_bar
        self.eps_p_i = self.eps_p + delta_eps_p

    def update(self):
        self.r = self.r_i
        self.eps_p = self.eps_p_i

    def calc_f_ip1(self, r: float, theta: float, del_gam: float, q_tri: float):
        return self.sig_y + theta * (r + self.b * self.Q * del_gam) + 3 * self.elastic.G * del_gam - q_tri

    def calc_f_ip1_prime(self, r: float, theta: float, del_gam: float):
        return 3 * self.elastic.G - self.b * theta**2 * (r + self.b * self.Q * del_gam) + theta * self.b * self.Q

    def calc_Dep(self, q_tri: float, n_bar: np.ndarray, f_ip1_prime: float, del_gam: float):
         return (
             self.elastic.De -
             6 * self.elastic.G**2 * del_gam / q_tri * Id +
             6 * self.elastic.G**2 * (del_gam / q_tri - 1 / f_ip1_prime) * np.outer(n_bar, n_bar)
         )

class Isotropic_voce_s:
    def __init__(self, elastic: Elastic, sig_y: float, Q: float, b: float):
        self.elastic = elastic
        self.sig_y = sig_y
        self.Q = Q
        self.b = b

    def return_mapping(self, eps: np.ndarray, del_eps: np.ndarray, eps_p: np.ndarray, r: float, eff_eps_p: float):
        print(f"Initial dela eps: {del_eps}")
        for itr in range(NW_I):
            print("-"*80)
            print(f"Iteration: {itr+1}")
            eps_tri = eps + del_eps
            eps_e_tri = eps_tri - eps_p
            sig_tri = self.elastic.De @ eps_e_tri
            sig_d_tri = Id @ sig_tri
            sig_v_tri = 1 / 3 * IxI @ sig_tri

            q_tri = np.sqrt(3/2 * sig_d_tri @ sig_d_tri)
            n_bar = sig_d_tri / q_tri
            del_gam = 0.0
            theta = 1.0

            f_ip1 = self.calc_f_ip1(r, theta, del_gam, q_tri)
            f_ip1_prime = self.calc_f_ip1_prime(r, theta, del_gam)
            if f_ip1 < 0.0:
                print("Plastic behavior")
                for inew in range(RM_I):
                    print(f"Return map iteration {inew+1}")
                    print(f_ip1)
                    d_del_gam = f_ip1 / f_ip1_prime
                    del_gam -= d_del_gam
                    theta = 1 / (1 + self.b * del_gam)
                    f_ip1 = self.calc_f_ip1(r, theta, del_gam, q_tri)
                    f_ip1_prime = self.calc_f_ip1_prime(r, theta, del_gam)
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
                r_ip1 = theta * (r + self.b * self.Q * del_gam)
                eff_eps_p_ip1 = eff_eps_p + del_gam
                sig_r = 2 * self.elastic.G * eps_e_d_ip1 + sig_v_tri
                Dep = self.calc_Dep(q_tri, n_bar, f_ip1_prime, del_gam)
            else:
                print("Elastic behavior")
                eps_e_ip1 = eps_e_tri
                eps_p_ip1 = eps_tri - eps_e_tri
                r_ip1 = r
                eff_eps_p_ip1 = eff_eps_p
                sig_r = self.elastic.De @ eps_e_ip1
                Dep = self.elastic.De
            return sig_r, eps_p_ip1, r_ip1, eff_eps_p_ip1, Dep

    def calc_f_ip1(self, r: float, theta: float, del_gam: float, q_tri: float):
        return self.sig_y + theta * (r + self.b * self.Q * del_gam) + 3 * self.elastic.G * del_gam - q_tri

    def calc_f_ip1_prime(self, r: float, theta: float, del_gam: float):
        return 3 * self.elastic.G - self.b * theta**2 * (r + self.b * self.Q * del_gam) + theta * self.b * self.Q

    def calc_Dep(self, q_tri: float, n_bar: np.ndarray, f_ip1_prime: float, del_gam: float):
         return (
             self.elastic.De -
             6 * self.elastic.G**2 * del_gam / q_tri * Id +
             6 * self.elastic.G**2 * (del_gam / q_tri - 1 / f_ip1_prime) * np.outer(n_bar, n_bar)
         )


class Stress_calculator:
    def __init__(self, hardening: Isotropic_voce_s, total_inc: int, delta=1.0):
        self.hardening = hardening
        self.total_inc = total_inc
        self.delta = delta
        self.initialize()

    def initialize(self):
        self.sig = np.array([0.0]*6)
        self.eps = np.array([0.0]*6)
        self.eps_p = np.array([0.0]*6)
        self.eff_eps_p = 0.0
        self.r = 0.0
        self.mises = 0.0

        self.sig_hist = []
        self.eps_hist = []
        self.eps_p_hist = []
        self.eff_eps_p_hist = []
        self.r_hist = []
        self.mises_hist = []

    def add_all(self):
        self.sig_hist.append(self.sig)
        self.eps_hist.append(self.eps)
        self.eps_p_hist.append(self.eps_p)
        self.eff_eps_p_hist.append(self.eff_eps_p)
        self.r_hist.append(self.r)
        self.mises_hist.append(self.mises)

    def calc_steps(self):
        self.initialize()
        for i in range(self.total_inc):
            print("="*80)
            print(f"Increment {i}")
            del_sig = np.array([self.delta, 0.0, 0.0, 0.0, 0.0, 0.0])
            sig_goal = self.sig + del_sig
            self.calc_stress(sig_goal)
            self.add_all()

    def calc_stress(self, sig_goal):
        f_sig = sig_goal - self.sig
        del_eps = np.linalg.inv(self.hardening.elastic.De) @ f_sig
        print(f"Initial dela eps: {del_eps}")

        for itr in range(NW_I):
            print("-"*80)
            print(f"Iteration: {itr+1}")
            sig_r, eps_p_ip1, r_ip1, eff_eps_p_ip1, Dep = self.hardening.return_mapping(self.eps, del_eps, self.eps_p, self.r, self.eff_eps_p)

            sig_diff = sig_goal - sig_r
            print(f"Corrected sig: {sig_r}")
            print(f"Difference norm: {np.sqrt(sig_diff @ sig_diff)}")
            if np.sqrt(3 / 2 * sig_diff @ sig_diff) / (self.hardening.sig_y + self.r) < TOL:
                self.sig = sig_goal
                self.eps_p = eps_p_ip1
                self.eps = self.eps + del_eps
                self.r = r_ip1
                self.eff_eps_p = eff_eps_p_ip1
                sig_d = Id @ self.sig
                self.mises = np.sqrt(3/2 * sig_d @ sig_d)
                print(f"Iteration converged: itr.{itr+1}")
                print(f"Mises: {self.mises}")
                break
            else:
                print(f"itr.{itr+1}")
                d_del_eps = np.linalg.inv(Dep) @ sig_diff
                del_eps += d_del_eps
                print(f"Updated delta eps: {del_eps}")
            if itr == NW_I - 1:
                raise ValueError(f"Not converged this iteration.")



E = 205000.0
n = 0.3
sig_y = 235
Q = 300.0
b = 150.0

elastic = Elastic(E, n)
iso01 = Isotropic_voce_s(elastic, sig_y, Q,  b)

calculator = Stress_calculator(iso01, 600)
calculator.calc_steps()

from matplotlib import pyplot as plt

fig = plt.figure()
plt.plot(calculator.eff_eps_p_hist, calculator.mises_hist)
plt.show()

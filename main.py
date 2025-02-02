import numpy as np

from util import Is, Id, IxI, Id_s

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


class Material_expression_base:
    def __init__(self, elastic: Elastic, sig_y: float):
        self.elastic = elastic
        self.sig_y = sig_y
        self.eps_p = np.array([0.0]*6)
        self.eff_eps_p = 0.0

    @property
    def yield_stress(self):
        return self.sig_y

    def initialize(self):
        raise NotImplementedError()

    def update_i(self):
        raise NotImplementedError()

    def update(self):
        raise NotImplementedError()

    def calc_f_ip1(self):
        raise NotImplementedError()

    def calc_f_ip1_prime(self):
        raise NotImplementedError()

    def calc_Dep(self):
        raise NotImplementedError()


class Linear_isotropic(Material_expression_base):
    def __init__(self, elastic: Elastic, sig_y: float, h: float):
        super().__init__(elastic, sig_y)
        self.h = h

    @property
    def yield_stress(self):
        return self.sig_y + self.r

    def initialize(self):
        self.r = 0.0
        self.r_i = 0.0
        self.eps_p = np.array([0.0]*6)
        self.eps_p_i = np.array([0.0]*6)
        self.eff_eps_p = 0.0
        self.eff_eps_p_i = 0.0

    def update_i(self, del_gam, n_bar):
        self.r_i = self.r + self.h * del_gam
        delta_eps_p = np.sqrt(3 / 2) * del_gam * n_bar * np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        self.eps_p_i = self.eps_p + delta_eps_p
        self.eff_eps_p_i = self.eff_eps_p + del_gam

    def update(self):
        self.r = self.r_i
        self.eps_p = self.eps_p_i
        self.eff_eps_p = self.eff_eps_p_i

    def calc_f_ip1(self, q_tri: float, del_gam: float):
        return self.sig_y + (self.r + self.h * del_gam) + 3 * self.elastic.G * del_gam - q_tri

    def calc_f_ip1_prime(self, del_gam: float):
        return 3 * self.elastic.G + self.h

    def calc_Dep(self, q_tri: float, del_gam: float, n_bar: np.ndarray):
         if del_gam == 0.0:
             return self.elastic.De
         f_ip1_prime = self.calc_f_ip1_prime(del_gam)
         return (
             self.elastic.De -
             6 * self.elastic.G**2 * del_gam / q_tri * Id +
             6 * self.elastic.G**2 * (del_gam / q_tri - 1 / f_ip1_prime) * np.outer(n_bar, n_bar)
         )


class Voce_isotropic(Material_expression_base):
    def __init__(self, elastic: Elastic, sig_y: float, Q: float, b: float):
        super().__init__(elastic, sig_y)
        self.Q = Q
        self.b = b

    @property
    def yield_stress(self):
        return self.sig_y + self.r

    def initialize(self):
        self.r = 0.0
        self.r_i = 0.0
        self.eps_p = np.array([0.0]*6)
        self.eps_p_i = np.array([0.0]*6)
        self.eff_eps_p = 0.0
        self.eff_eps_p_i = 0.0

    def update_i(self, del_gam, n_bar):
        theta = 1 / (1 + self.b * del_gam)
        self.r_i = theta * (self.r + self.b * self.Q * del_gam)
        delta_eps_p = np.sqrt(3 / 2) * del_gam * n_bar * np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        self.eps_p_i = self.eps_p + delta_eps_p
        self.eff_eps_p_i = self.eff_eps_p + del_gam

    def update(self):
        self.r = self.r_i
        self.eps_p = self.eps_p_i
        self.eff_eps_p = self.eff_eps_p_i

    def calc_f_ip1(self, q_tri: float, del_gam: float):
        theta = 1 / (1 + self.b * del_gam)
        return self.sig_y + theta * (self.r + self.b * self.Q * del_gam) + 3 * self.elastic.G * del_gam - q_tri

    def calc_f_ip1_prime(self, del_gam: float):
        theta = 1 / (1 + self.b * del_gam)
        return 3 * self.elastic.G - self.b * theta**2 * (self.r + self.b * self.Q * del_gam) + theta * self.b * self.Q

    def calc_Dep(self, q_tri: float, del_gam: float, n_bar: np.ndarray):
         if del_gam == 0.0:
             return self.elastic.De
         f_ip1_prime = self.calc_f_ip1_prime(del_gam)
         return (
             self.elastic.De -
             6 * self.elastic.G**2 * del_gam / q_tri * Id +
             6 * self.elastic.G**2 * (del_gam / q_tri - 1 / f_ip1_prime) * np.outer(n_bar, n_bar)
         )


class Calculator_base:
    def __init__(self, material: Material_expression_base, goal_sig: np.ndarray, step: int):
        self.material = material
        self.goal_sig = goal_sig
        self.step = step
        self.material.initialize()
        self.eps = np.array([0.0]*6)
        self.sig = np.array([0.0]*6)
        self.output = Output_data()

    @staticmethod
    def calc_stress_norm(sig):
        r = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        sig_r = sig * r
        return np.sqrt(sig @ sig_r)

    def initialize(self):
        self.eps = np.array([0.0]*6)
        self.sig = np.array([0.0]*6)
        self.output.initialize()

    def return_mapping(self, q_tri, n_bar):
        for itr in range(NW_I):
            print("-"*80)
            print(f"Return map iteration: {itr+1}")
            del_gam = 0.0

            f_ip1 = self.material.calc_f_ip1(q_tri, del_gam)
            f_ip1_prime = self.material.calc_f_ip1_prime(del_gam)
            if f_ip1 < 0.0:
                print("Plastic behavior")
                for inew in range(RM_I):
                    print(f"Return map iteration {inew+1}")
                    d_del_gam = f_ip1 / f_ip1_prime
                    del_gam -= d_del_gam
                    f_ip1 = self.material.calc_f_ip1(q_tri, del_gam)
                    f_ip1_prime = self.material.calc_f_ip1_prime(del_gam)
                    if abs(f_ip1) < TOL:
                        if del_gam < 0.0:
                            raise ValueError("Delta gamma is negative value.")
                        print(f"Return map converged itr.{inew+1}")
                        print(f"Delta gamma: {del_gam}")
                        break
                    if inew == RM_I - 1:
                        raise ValueError("Return map isn't converged")
            else:
                print("Elastic behavior")
            return del_gam

    def integrate_stress(self, del_eps):
        eps_tri = self.eps + del_eps
        eps_e_tri = eps_tri - self.material.eps_p
        sig_tri = self.material.elastic.De @ eps_e_tri
        sig_d_tri = Id_s @ sig_tri
        sig_v_tri = 1 / 3 * IxI @ sig_tri

        q_tri = np.sqrt(3 / 2) * self.calc_stress_norm(sig_d_tri)
        n_bar = sig_d_tri / (q_tri / np.sqrt(3 / 2))
        print(f"N_norm: {self.calc_stress_norm(n_bar)}")
        del_gam = self.return_mapping(q_tri, n_bar)
        print(f"F: {self.material.calc_f_ip1(q_tri, del_gam)}")
        self.material.update_i(del_gam, n_bar)
        eps_e_i = eps_tri - self.material.eps_p_i
        sig_e = self.material.elastic.De @ eps_e_i
        sig_d = Id_s @ sig_e
        sig = sig_d + sig_v_tri
        Dep = self.material.calc_Dep(q_tri, del_gam, n_bar)
        return sig, Dep

    def calc_increment(self, goal):
        f_sig = goal - self.sig
        del_eps = np.linalg.inv(self.material.elastic.De) @ f_sig
        print(f"Goal sig: {goal}")
        print(f"Current sig: {self.sig}")
        print(f"Initial dela eps: {del_eps}")
        for itr in range(NW_I):
            print("-"*80)
            print(f"Iteration: {itr+1}")
            sig_i, Dep = self.integrate_stress(del_eps)
            d_sig = Dep @ del_eps
            print(f"Corrected sig: {sig_i}")
            print(f"Corrected sig2: {self.sig + d_sig}")
            sig_diff = goal - sig_i
            print(f"{sig_diff}")
            sig_diff_norm = self.calc_stress_norm(sig_diff)
            print(f"Difference norm: {sig_diff_norm}")
            if np.sqrt(3 / 2) * sig_diff_norm / self.material.yield_stress < TOL:
                self.sig = goal
                self.eps = self.eps + del_eps
                self.material.update()
                print(f"Iteration converged: itr.{itr+1}")
                break
            else:
                print(f"itr.{itr+1}")
                d_del_eps = np.linalg.inv(Dep) @ sig_diff
                del_eps += d_del_eps
                print(f"Updated delta eps: {del_eps}")
            if itr == NW_I - 1:
                raise ValueError(f"Not converged this iteration.")

    def calculate_steps(self, is_init=True):
        if is_init:
            self.initialize()
        for inc in range(self.step):
            print("="*80)
            print(f"Increment {inc}")
            goal = (inc + 1) / self.step * self.goal_sig
            self.calc_increment(goal)
            self.output.add_data(self.sig, self.eps, self.material.eps_p, self.material.eff_eps_p)


class Output_data:
    def __init__(self):
        self.initialize()

    def initialize(self):
        self.sig = []
        self.eps = []
        self.eps_p = []
        self.eff_eps_p = []
        self.mises = []

    def add_data(self, sig, eps, eps_p, eff_eps_p):
        self.sig.append(sig)
        self.eps.append(eps)
        self.eps_p.append(eps_p)
        self.eff_eps_p.append(eff_eps_p)
        self.mises.append(self.calc_mises(sig))

    def calc_mises(self, sig):
        sig_d = Id_s @ sig
        return np.sqrt(3 / 2) * self.calc_stress_norm(sig_d)

    @staticmethod
    def calc_stress_norm(sig):
        r = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        sig_r = sig * r
        return np.sqrt(sig @ sig_r)


E = 205000.0
n = 0.3
sig_y = 235
Q = 300.0
b = 150.0

elastic = Elastic(E, n)
linear_iso = Linear_isotropic(elastic, sig_y, 20000.0)
voce_iso = Voce_isotropic(elastic, sig_y, Q, b)
calculator = Calculator_base(voce_iso, np.array([0.0, 0.0, 0.0, 150.0, 0.0, 0.0]), 400)
calculator.calculate_steps()
calculator.goal_sig = np.array([0.0, 0.0, 0.0, -180.0, 0.0, 0.0])
calculator.calculate_steps(is_init=False)
calculator.goal_sig = np.array([0.0, 0.0, 0.0, 200.0, 0.0, 0.0])
calculator.calculate_steps(is_init=False)

x = [e[3] for e in calculator.output.eps_p]
y = [s[3] for s in calculator.output.sig]


from matplotlib import pyplot as plt

fig = plt.figure()
plt.plot(x, y)
plt.show()

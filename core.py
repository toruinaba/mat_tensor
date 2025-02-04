import numpy as np
from util import IxI, Id_s
from material import Material_expression_base


class Calculator3D:
    TOL = 1.0e-06
    RM_I = 10
    NW_I = 100

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
        for itr in range(self.NW_I):
            print("-"*80)
            print(f"Return map iteration: {itr+1}")
            del_gam = 0.0

            f_ip1 = self.material.calc_f_ip1(q_tri, del_gam, n_bar)
            f_ip1_prime = self.material.calc_f_ip1_prime(q_tri, del_gam, n_bar)
            if f_ip1 < 0.0:
                print("Plastic behavior")
                for inew in range(self.RM_I):
                    print(f"Return map iteration {inew+1}")
                    d_del_gam = f_ip1 / f_ip1_prime
                    del_gam -= d_del_gam
                    f_ip1 = self.material.calc_f_ip1(q_tri, del_gam, n_bar)
                    f_ip1_prime = self.material.calc_f_ip1_prime(q_tri, del_gam, n_bar)
                    if abs(f_ip1) < self.TOL:
                        if del_gam < 0.0:
                            raise ValueError("Delta gamma is negative value.")
                        print(f"Return map converged itr.{inew+1}")
                        print(f"Delta gamma: {del_gam}")
                        break
                    if inew == self.RM_I - 1:
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

        q_tri, n_bar = self.material.calc_tri(sig_d_tri)
        del_gam = self.return_mapping(q_tri, n_bar)
        print(f"F: {self.material.calc_f_ip1(q_tri, del_gam, n_bar)}")
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
        for itr in range(self.NW_I):
            print("-"*80)
            print(f"Iteration: {itr+1}")
            sig_i, Dep = self.integrate_stress(del_eps)
            print(f"Corrected sig: {sig_i}")
            sig_diff = goal - sig_i
            sig_diff_norm = self.calc_stress_norm(sig_diff)
            print(f"Difference norm: {sig_diff_norm}")
            if np.sqrt(3 / 2) * sig_diff_norm / self.material.yield_stress < self.TOL:
                self.sig = goal
                self.eps = self.eps + del_eps
                self.material.update()
                print(f"Stress integration converged: itr.{itr+1}")
                break
            else:
                print(f"itr.{itr+1}")
                d_del_eps = np.linalg.inv(Dep) @ sig_diff
                del_eps += d_del_eps
                print(f"Updated delta eps: {del_eps}")
            if itr == self.NW_I - 1:
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


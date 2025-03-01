from copy import deepcopy
import numpy as np
from src.util import IxI, Id_s
from src.material import Material_expression_base


class Calculator3D:
    TOL = 1.0e-06
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

    def calc_increment(self, goal):
        f_sig = goal - self.sig
        del_eps = np.linalg.inv(self.material.elastic.De) @ f_sig
        print(f"Goal sig: {goal}")
        print(f"Current sig: {self.sig}")
        print(f"Initial dela eps: {del_eps}")
        for itr in range(self.NW_I):
            print("-"*80)
            print(f"Iteration: {itr+1}")
            sig_i, Dep = self.material.integrate_stress(self.eps, del_eps)
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
        initial_sig = deepcopy(self.sig)
        for inc in range(self.step):
            print("="*80)
            print(f"Increment {inc}")
            goal = (inc + 1) / self.step * (self.goal_sig - initial_sig) + initial_sig
            print(f"goal: {goal}")
            self.calc_increment(goal)
            self.output.add_data(self.sig, self.eps, self.material.eps_p, self.material.eff_eps_p, self.material.theta, self.material.beta, self.material.r, self.material.R, self.material.q)


class Output_data:
    def __init__(self):
        self.initialize()

    def initialize(self):
        self.sig = []
        self.eps = []
        self.eps_p = []
        self.eff_eps_p = []
        self.mises = []
        self.r = []
        self.R = []
        self.q = []
        self.theta = []
        self.beta = []
        self.gr = []

    def add_data(self, sig, eps, eps_p, eff_eps_p, theta, beta, r, R, q):
        self.sig.append(sig)
        self.eps.append(eps)
        self.eps_p.append(eps_p)
        self.eff_eps_p.append(eff_eps_p)
        self.mises.append(self.calc_mises(sig))
        self.theta.append(theta)
        self.beta.append(beta)
        self.r.append(r)
        self.R.append(R)
        self.q.append(q)
        self.gr.append(self.calc_mises(beta - q))

    def calc_mises(self, sig):
        sig_d = Id_s @ sig
        return np.sqrt(3 / 2) * self.calc_stress_norm(sig_d)

    @staticmethod
    def calc_stress_norm(sig):
        r = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        sig_r = sig * r
        return np.sqrt(sig @ sig_r)


from copy import deepcopy
import numpy as np
import logging
from src.solid.utils import Id_s
from src.solid.materials import Material_expression_base
from src.error import NotConvergedError


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Calculator3D:
    TOL = 1.0e-06
    NW_I = 20
    TOTAL_TIME = 1.0
    MAX_INCREMENT = 1000

    def __init__(self, material: Material_expression_base, goal_sig: np.ndarray, init_delta_t=0.01, min_delta_t=1.0e-05, max_delta_t=0.01):
        self.material = material
        self.goal_sig = goal_sig
        self.init_delta_t = init_delta_t
        self.min_delta_t = min_delta_t
        self.max_delta_t = max_delta_t
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
        self.current_time = 0.0
        self.current_delta_t = self.init_delta_t
        self.current_inc = 0
        self.output.initialize()

    def calc_increment(self, goal):
        f_sig = goal - self.sig
        del_eps = np.linalg.inv(self.material.elastic.De) @ f_sig
        logger.info(f"Goal sig: {goal}")
        logger.info(f"Current sig: {self.sig}")
        logger.debug(f"Initial dela eps: {del_eps}")
        for s_itr in range(self.NW_I):
            logger.debug(f"Iteration: {s_itr+1}")
            sig_i, Dep = self.material.integrate_stress(self.eps, del_eps)
            logger.debug(f"Corrected sig: {sig_i}")
            sig_diff = goal - sig_i
            sig_diff_norm = self.calc_stress_norm(sig_diff)
            logger.debug(f"Difference norm: {sig_diff_norm}")
            if np.sqrt(3 / 2) * sig_diff_norm / self.material.yield_stress < self.TOL:
                self.sig = goal
                self.eps = self.eps + del_eps
                self.material.update()
                logger.info(f"Stress integration converged: itr.{s_itr+1}")
                break
            else:
                logger.debug(f"itr.{s_itr+1}")
                d_del_eps = np.linalg.inv(Dep) @ sig_diff
                del_eps += d_del_eps
                logger.debug(f"Updated delta eps: {del_eps}")
        else:
            raise NotConvergedError("Stress integration isn't converged")

    def calculate_steps(self, is_init=True):
        counter = 0
        if is_init:
            self.initialize()
        initial_sig = deepcopy(self.sig)
        self.current_time = 0.0
        self.current_delta_t = self.init_delta_t
        self.current_inc = 0
        while self.current_time < 1.0:
            self.current_inc += 1
            logger.info(f"Increment {self.current_inc}")
            attempt_time = self.current_time + self.current_delta_t if self.current_time + self.current_delta_t < 1.0 else 1.0
            logger.info(f"Attempt time: {attempt_time}")
            goal = attempt_time / self.TOTAL_TIME * (self.goal_sig - initial_sig) + initial_sig
            try:
                self.calc_increment(goal)
                counter += 1
                self.current_time = attempt_time
                if counter == 4:
                    self.current_delta_t = self.current_delta_t * 1.25 if self.current_delta_t * 1.25 < self.max_delta_t else self.max_delta_t
                    counter = 0
            except NotConvergedError:
                if self.current_delta_t < self.min_delta_t:
                    raise NotConvergedError("Minimum time increment is exceeded.")
                logger.warning("Not converged. Try smaller time increment.")
                self.current_delta_t *= 0.25
            if self.current_inc >= self.MAX_INCREMENT:
                raise NotConvergedError("Maximum increment is exceeded.")
            self.output.add_data(self.sig, self.eps, self.material.eps_p, self.material.eff_eps_p)
            logger.info(f"Ended time: {self.current_time}")


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


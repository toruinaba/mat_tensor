import numpy as np
import logging
from src.axisymmetry.utils import I_ax, Is_ax, Id_ax, Id_s_ax, IxI_ax
from src.error import NotConvergedError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Elastic_ax:
    def __init__(self, E, n):
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
        return 2 * self.G * Is_ax + 1 * (self.K - 2 / 3 * self.G) * IxI_ax

    @property
    def De_inv(self):
        return np.linalg.inv(self.De)


class Material_expression_base_ax:
    TOL = 1.0e-06
    RM_I = 10

    def __init__(self, elastic: Elastic_ax, sig_y: float):
        self.elastic = elastic
        self.sig_y = sig_y
        self.eps = np.zeros(4)
        self.eps_p = np.zeros(4)
        self.eps_p_i = np.zeros(4)
        self.eff_eps_p = 0.0
        self.eff_eps_p_i = 0.0

    @staticmethod
    def calc_stress_norm(sig):
        r = np.array([1.0, 1.0, 2.0, 1.0])
        sig_r = sig * r
        return np.sqrt(sig @ sig_r)

    @property
    def yield_stress(self):
        return self.sig_y

    def initialize(self):
        raise NotImplementedError()

    def update_i(self):
        raise NotImplementedError()

    def update(self):
        raise NotImplementedError()

    def calc_tri(self, sig_d, del_gam):
        raise NotImplementedError()

    def calc_f_ip1(self):
        raise NotImplementedError()

    def calc_f_ip1_prime(self):
        raise NotImplementedError()

    def calc_Dep(self):
        raise NotImplementedError()
    
    def return_mapping(self, sig_d_tri):
        del_gam = 0.0
        f_ip1 = self.calc_f_ip1(sig_d_tri, del_gam)
        f_ip1_prime = self.calc_f_ip1_prime(sig_d_tri, del_gam)
        if f_ip1 < 0.0:
            logger.debug("Plastic behavior")
            for inew in range(self.RM_I):
                logger.debug(f"Newton iteration {inew+1}")
                d_del_gam = f_ip1 / f_ip1_prime
                del_gam -= d_del_gam
                f_ip1 = self.calc_f_ip1(sig_d_tri, del_gam)
                f_ip1_prime = self.calc_f_ip1_prime(sig_d_tri, del_gam)
                if abs(f_ip1) < self.TOL:
                    if del_gam < 0.0:
                        raise ValueError("Delta gamma is negative value.")
                    logger.info(f"Return map converged itr.{inew+1}")
                    logger.debug(f"Delta gamma: {del_gam}")
                    break
            else:
                raise NotConvergedError("Return map isn't converged")
        else:
            logger.debug("Elastic behavior")
        q_tri, n_bar = self.calc_tri(sig_d_tri, del_gam)
        return q_tri, del_gam, n_bar

    def integrate_stress(self, eps, del_eps):
        eps_tri = eps + del_eps
        eps_e_tri = eps_tri - self.eps_p
        sig_tri = self.elastic.De @ eps_e_tri
        sig_d_tri = Id_s_ax @ sig_tri

        q_tri, del_gam, n_bar = self.return_mapping(sig_d_tri)
        logger.debug(f"F: {self.calc_f_ip1(sig_d_tri, del_gam)}")
        self.update_i(del_gam, n_bar)
        eps_e_i = eps_tri - self.eps_p_i
        sig_e = self.elastic.De @ eps_e_i
        sig = sig_tri - 2 * self.elastic.G * del_gam * n_bar * np.sqrt(3 / 2)
        Dep = self.calc_Dep(sig_d_tri, del_gam)
        return sig, Dep


class Linear_isotropic_ax(Material_expression_base_ax):
    def __init__(self, elastic: Elastic_ax, sig_y: float, h: float):
        super().__init__(elastic, sig_y)
        self.h = h

    @property
    def yield_stress(self):
        return self.sig_y + self.r

    def initialize(self):
        self.r = 0.0
        self.r_i = 0.0

    def update_i(self, del_gam, n_bar):
        self.r_i = self.r + self.h * del_gam
        delta_eps_p = np.sqrt(3 / 2) * del_gam * n_bar * np.array([1.0, 1.0, 2.0, 1.0])
        self.eps_p_i = self.eps_p + delta_eps_p
        self.eff_eps_p_i = self.eff_eps_p + del_gam

    def update(self):
        self.r = self.r_i
        self.eps_p = self.eps_p_i
        self.eff_eps_p = self.eff_eps_p_i

    def calc_tri(self, sig_d, del_gam):
        norm = self.calc_stress_norm(sig_d)
        return np.sqrt(3 / 2) * norm, sig_d / norm

    def calc_f_ip1(self, sig_d: np.ndarray, del_gam: float):
        q_tri, n_bar = self.calc_tri(sig_d, del_gam)
        return self.sig_y + (self.r + self.h * del_gam) + 3 * self.elastic.G * del_gam - q_tri

    def calc_f_ip1_prime(self, sig_d: np.ndarray, del_gam: float):
        return 3 * self.elastic.G + self.h

    def calc_Dep(self, sig_d: np.ndarray, del_gam: float):
         q_tri, n_bar = self.calc_tri(sig_d, del_gam)
         if del_gam == 0.0:
             return self.elastic.De
         f_ip1_prime = self.calc_f_ip1_prime(sig_d, del_gam)
         return (
             self.elastic.De -
             6 * self.elastic.G**2 * del_gam / q_tri * Id_ax +
             6 * self.elastic.G**2 * (del_gam / q_tri - 1 / f_ip1_prime) * np.outer(n_bar, n_bar)
         )


class Linear_kinematic_ax(Material_expression_base_ax):
    def __init__(self, elastic: Elastic_ax, sig_y: float, h: float):
        super().__init__(elastic, sig_y)
        self.h = h

    def initialize(self):
        self.alpha = np.zeros(4)
        self.alpha_i = np.zeros(4)
        self.eps_p = np.zeros(4)
        self.eps_p_i = np.zeros(4)
        self.eff_eps_p = 0.0
        self.eff_eps_p_i = 0.0

    def update_i(self, del_gam, n_bar):
        delta_eps_p = np.sqrt(3 / 2) * del_gam * n_bar * np.array([1.0, 1.0, 2.0, 1.0])
        delta_alpha = 2 / 3 * delta_eps_p * self.h
        self.alpha_i = self.alpha + delta_alpha
        self.eps_p_i = self.eps_p + delta_eps_p
        self.eff_eps_p_i = self.eff_eps_p + del_gam

    def update(self):
        self.eps_p = self.eps_p_i
        self.alpha = self.alpha_i
        self.eff_eps_p = self.eff_eps_p_i

    def calc_tri(self, sig_d, del_gam):
        eta = sig_d - self.alpha
        norm = self.calc_stress_norm(eta)
        return np.sqrt(3 / 2) * norm, eta / norm

    def calc_f_ip1(self, sig_d: np.ndarray, del_gam: float):
        q_tri, n_bar = self.calc_tri(sig_d, del_gam)
        return self.sig_y + 3 * self.elastic.G * del_gam + self.h * del_gam - q_tri

    def calc_f_ip1_prime(self, sig_d: np.ndarray, del_gam: float):
        return 3 * self.elastic.G + self.h

    def calc_Dep(self, sig_d: np.ndarray, del_gam: float):
         q_tri, n_bar = self.calc_tri(sig_d, del_gam)
         if del_gam == 0.0:
             return self.elastic.De
         f_ip1_prime = self.calc_f_ip1_prime(sig_d, del_gam)
         return (
             self.elastic.De -
             6 * self.elastic.G**2 * del_gam / q_tri * Id_ax +
             6 * self.elastic.G**2 * (del_gam / q_tri - 1 / f_ip1_prime) * np.outer(n_bar, n_bar)
         )


class Voce_isotropic_ax(Material_expression_base_ax):
    def __init__(self, elastic: Elastic_ax, sig_y: float, Q: float, b: float):
        super().__init__(elastic, sig_y)
        self.Q = Q
        self.b = b

    @property
    def yield_stress(self):
        return self.sig_y + self.r

    def initialize(self):
        self.r = 0.0
        self.r_i = 0.0
        self.eps_p = np.zeros(4)
        self.eps_p_i = np.zeros(4)
        self.eff_eps_p = 0.0
        self.eff_eps_p_i = 0.0

    def update_i(self, del_gam, n_bar):
        theta = 1 / (1 + self.b * del_gam)
        self.r_i = theta * (self.r + self.b * self.Q * del_gam)
        delta_eps_p = np.sqrt(3 / 2) * del_gam * n_bar * np.array([1.0, 1.0, 2.0, 1.0])
        self.eps_p_i = self.eps_p + delta_eps_p
        self.eff_eps_p_i = self.eff_eps_p + del_gam

    def update(self):
        self.r = self.r_i
        self.eps_p = self.eps_p_i
        self.eff_eps_p = self.eff_eps_p_i

    def calc_tri(self, sig_d: np.ndarray, del_gam: float):
        norm = self.calc_stress_norm(sig_d)
        return np.sqrt(3 / 2) * norm, sig_d / norm

    def calc_f_ip1(self, sig_d: np.ndarray, del_gam: float):
        q_tri, n_bar = self.calc_tri(sig_d, del_gam)
        theta = 1 / (1 + self.b * del_gam)
        return self.sig_y + theta * (self.r + self.b * self.Q * del_gam) + 3 * self.elastic.G * del_gam - q_tri

    def calc_f_ip1_prime(self, sig_d: np.ndarray, del_gam: float):
        theta = 1 / (1 + self.b * del_gam)
        return 3 * self.elastic.G - self.b * theta**2 * (self.r + self.b * self.Q * del_gam) + theta * self.b * self.Q

    def calc_Dep(self, sig_d: np.ndarray, del_gam: float):
         q_tri, n_bar = self.calc_tri(sig_d, del_gam)
         if del_gam == 0.0:
             return self.elastic.De
         f_ip1_prime = self.calc_f_ip1_prime(sig_d, del_gam)
         return (
             self.elastic.De -
             6 * self.elastic.G**2 * del_gam / q_tri * Id_ax +
             6 * self.elastic.G**2 * (del_gam / q_tri - 1 / f_ip1_prime) * np.outer(n_bar, n_bar)
         )


class AF_kinematic_ax(Material_expression_base_ax):
    def __init__(self, elastic: Elastic_ax, sig_y: float, C: float, k: float):
        super().__init__(elastic, sig_y)
        self.C = C
        self.k = k

    def initialize(self):
        self.alpha = np.zeros(4)
        self.alpha_i = np.zeros(4)
        self.eps_p = np.zeros(4)
        self.eps_p_i = np.zeros(4)
        self.eff_eps_p = 0.0
        self.eff_eps_p_i = 0.0

    def update_i(self, del_gam, n_bar):
        delta_eps_p = np.sqrt(3 / 2) * del_gam * np.diag([1.0, 1.0, 2.0, 1.0]) @ n_bar
        theta = 1 / (1 + self.k * del_gam)
        self.alpha_i = theta * (self.alpha + np.sqrt(2 / 3) * self.C * del_gam * n_bar)
        self.eps_p_i = self.eps_p + delta_eps_p
        self.eff_eps_p_i = self.eff_eps_p + del_gam

    def update(self):
        self.eps_p = self.eps_p_i
        self.alpha = self.alpha_i
        self.eff_eps_p = self.eff_eps_p_i

    def calc_tri(self, sig_d, del_gam):
        theta = 1 / (1 + self.k * del_gam)
        eta = sig_d - self.alpha * theta
        norm = self.calc_stress_norm(eta)
        return np.sqrt(3 / 2) * norm, eta / norm

    def calc_f_ip1(self, sig_d: np.ndarray, del_gam: float):
        q_tri, n_bar = self.calc_tri(sig_d, del_gam)
        theta = 1 / (1 + self.k * del_gam)
        return self.sig_y + 3 * self.elastic.G * del_gam + self.C * theta * del_gam - q_tri

    def calc_f_ip1_prime(self, sig_d: np.ndarray, del_gam: float):
        q_tri, n_bar = self.calc_tri(sig_d, del_gam)
        theta = 1 / (1 + self.k * del_gam)
        dotted = n_bar @ self.alpha
        return (
            3 * self.elastic.G +
            self.C * theta -
            self.k * self.C * theta**2 * del_gam
            - np.sqrt(3/2) * self.k * theta**2 * dotted
        )

    def calc_Dep(self, sig_d: np.ndarray, del_gam: float):
        q_tri, n_bar = self.calc_tri(sig_d, del_gam)
        if del_gam == 0.0:
            return self.elastic.De
        f_ip1_prime = self.calc_f_ip1_prime(sig_d, del_gam)
        theta = 1 / (1 + self.k * del_gam)
        N4d = I_ax - np.outer(n_bar, n_bar)
        N4d_alpha = N4d @ self.alpha
        
        return (
            self.elastic.De -
            6 * self.elastic.G**2 * del_gam / q_tri * Id_ax +
            6 * self.elastic.G**2 * (del_gam / q_tri - 1 / f_ip1_prime) * np.outer(n_bar, n_bar) -
            3 * np.sqrt(6) * self.elastic.G**2 * self.k * theta**2 * del_gam / (q_tri * f_ip1_prime) * np.outer(N4d_alpha, n_bar)
        )



class Yoshida_uemori_ax:
    TOL = 1.0e-06
    RM_I = 20
    Q = np.diag([1.0, 1.0, 2.0, 1.0])

    def __init__(self, elastic: Elastic_ax, sig_y: float, B, C, Rsat, k, b, h, Ea, psi):
        self.elastic = elastic
        self.sig_y = sig_y
        self.B = B
        self.C = C
        self.Rsat = Rsat
        self.k = k
        self.b = b
        self.h = h
        self.Ea = Ea
        self.psi = psi
        self.eps_p = np.zeros(4)
        self.eps_p_i = np.zeros(4)
        self.eff_eps_p = 0.0
        self.eff_eps_p_i = 0.0
        self.theta = np.zeros(4)
        self.theta_i = np.zeros(4)
        self.beta = np.zeros(4)
        self.beta_i = np.zeros(4)
        self.R = 0.0
        self.R_i = 0.0
        self.r = 0.0
        self.r_i = 0.0
        self.q = np.zeros(4)
        self.q_i = np.zeros(4)
        self.initialize()

    @staticmethod
    def calc_stress_norm(sig):
        r = np.array([1.0, 1.0, 2.0, 1.0])
        sig_r = sig * r
        return np.sqrt(sig @ sig_r)

    @staticmethod
    def calc_De_factor(Ea, E, psi, eff_eps_p):
        return 1.0 - (1.0 - Ea / E) * (1.0 - np.exp(-psi * eff_eps_p))

    @property
    def yield_stress(self):
        return self.sig_y

    @property
    def a(self):
        return self.B + self.R - self.sig_y

    @property
    def De_factor(self):
        return self.calc_De_factor(self.Ea, self.elastic.E, self.psi, self.eff_eps_p)

    @property
    def De(self):
        return self.De_factor * self.elastic.De

    @property
    def De_inv(self):
        return (1 / self.De_factor) * self.elastic.De_inv

    def initialize(self):
        self.eps_p = np.zeros(4)
        self.eps_p_i = np.zeros(4)
        self.eff_eps_p = 0.0
        self.eff_eps_p_i = 0.0
        self.theta = np.zeros(4)
        self.theta_i = np.zeros(4)
        self.beta = np.zeros(4)
        self.beta_i = np.zeros(4)
        self.r = 0.0
        self.r_i = 0.0
        self.q = np.zeros(4)
        self.q_i = np.zeros(4)

    def update_i(self, delta_beta, delta_theta, delta_gam, n_flow):
        self.beta_i = self.beta + delta_beta
        self.theta_i = self.theta + delta_theta
        self.eps_p_i = self.eps_p + delta_gam * n_flow
        self.eff_eps_p_i = self.eff_eps_p + delta_gam
        xi_n = self.beta_i - self.q
        g_stag = self.calc_g_stag(xi_n, self.r)
        g_stag_flow = (self.Q @ xi_n) @ delta_beta
        if g_stag > -self.TOL and g_stag_flow > -self.TOL:
            logger.info("Hardening evolution")
            xi = self.beta - self.q
            if abs(np.sqrt(self.calc_g_stag(xi, self.r))) < self.TOL:
                delta_beta_s = delta_beta
            else:
                xi_xi = (self.Q @ xi) @ xi
                dbeta_dbeta = (self.Q @ delta_beta) @ delta_beta
                xi_dbeta = (self.Q @ xi) @ delta_beta
                r_diff = (-3 * xi_dbeta + np.sqrt((3 * xi_dbeta)**2 - 3 * dbeta_dbeta * (3 * xi_xi - 2 * self.r**2))) / (3 * dbeta_dbeta)
                beta_s = xi + r_diff * delta_beta
                xi_s = beta_s - self.q
                delta_beta_s = (1 - r_diff) * delta_beta
            xi_P_del_beta = (self.Q @ xi_n) @ delta_beta_s
            xi_P_xi = (self.Q @ xi_n) @ xi_n
            if abs(self.r) < self.TOL:
                delta_mu = 3 * xi_P_xi / (6 * self.h * xi_P_del_beta) - 1
            else:
                s = (-3 * self.h * xi_P_del_beta + np.sqrt((3 * self.h * xi_P_del_beta)**2 + 4 * self.r**2 * 3 / 2 * xi_P_xi)) / (2 * self.r**2)
                delta_mu = s - 1
                if delta_mu < 0.0:
                    raise ValueError(f"Delta mu is negative({delta_mu})")
            xi_i = xi_n / (1 + delta_mu)
            q_dot_i = delta_mu * xi_i
            self.q_i = self.q + q_dot_i
            self.r_i = np.sqrt(self.r**2 + 3 * self.h * (self.Q @ xi_i) @ (delta_beta_s))
            self.R_i = 1 / (1 + self.k * delta_gam) * (self.R + self.k * self.Rsat * delta_gam)
            xi_last = self.beta_i - self.q_i
        else:
            logger.info("Hardening stagnation")
            self.q_i = self.q
            self.r_i = self.r
            self.R_i = self.R

    def update(self):
        self.eps_p = self.eps_p_i
        self.eff_eps_p = self.eff_eps_p_i
        self.beta = self.beta_i
        self.theta = self.theta_i
        self.R = self.R_i
        self.q = self.q_i
        self.r = self.r_i

    def calc_g(self, sig_d):
        norm = self.calc_stress_norm(sig_d)
        n_bar = sig_d / norm
        return np.sqrt(3 / 2) * norm, np.sqrt(3 / 2) * n_bar

    def calc_g_flow(self, sig_d):
        norm = self.calc_stress_norm(sig_d)
        n_bar = sig_d / norm
        return np.sqrt(3 / 2) * norm, np.sqrt(3 / 2) * self.Q @ n_bar

    def calc_g_stag(self, xi, r):
        return 3 / 2 * (self.Q @ xi) @ xi - r**2

    def calc_delta_mu(self, delta_beta):
        if self.r == 0:
            return 0.0
        xi_n = self.beta + delta_beta - self.q
        xi_P_del_beta = xi_n @ (Id_s @ delta_beta)
        xi_P_xi = xi_n @ (Id_s @ xi_n)
        return (3 * self.h * xi_P_del_beta + np.sqrt((3 * self.h * xi_P_del_beta)**2 + 6 * self.r**2 * xi_P_xi)) / (2 * self.r**2) - 1

    def calc_f_f(self, eta):
        g_eta, n_s = self.calc_g(eta)
        return g_eta - self.sig_y

    def calc_j_f_f(self, eta):
        g_eta, n_s_f = self.calc_g_flow(eta)
        f_f_dsig = n_s_f
        f_f_dbeta = - n_s_f
        f_f_dtheta = - n_s_f
        f_f_dgamma = [0.0]
        vectors = (f_f_dsig, f_f_dbeta, f_f_dtheta, f_f_dgamma)
        return np.hstack(vectors)

    def calc_f_ep(self, sig_d, sig_d_tri, eta, delta_gam):
        g_eta, n_s_f = self.calc_g_flow(eta)
        return (self.De_inv @ (sig_d - sig_d_tri)) + delta_gam * n_s_f

    def calc_j_f_ep(self, sig_d, sig_d_tri, eta, delta_gam):
        g_eta, n_s_f = self.calc_g_flow(eta)
        updated_factor = self.calc_De_factor(self.Ea, self.elastic.E, self.psi, self.eff_eps_p + delta_gam)
        dn_dsig = 3 /(2 * g_eta) * (I_ax - np.outer(n_s_f / np.sqrt(3 / 2), n_s_f / np.sqrt(3 / 2)))
        f_ep_dsig = ((1 / updated_factor) * self.elastic.De_inv + delta_gam * dn_dsig)
        f_ep_dbeta = - delta_gam * dn_dsig
        f_ep_dtheta = - delta_gam * dn_dsig
        factor = 1.0 - (1 - self.Ea / self.elastic.E) * (1 - np.exp(-self.psi * (self.eff_eps_p + delta_gam)))
        f_ep_dgamma = self.psi * (factor - self.Ea / self.elastic.E) / factor**2 * self.elastic.De_inv @ (sig_d - sig_d_tri) + n_s_f
        matrices = (f_ep_dsig, f_ep_dbeta, f_ep_dtheta, np.matrix(f_ep_dgamma).transpose())
        return np.hstack(matrices)

    def calc_f_beta(self, eta, beta, delta_gam):
        return beta - self.beta - (self.k * self.b / self.sig_y *  eta - self.k * (beta)) * delta_gam

    def calc_j_f_beta(self, eta, beta, delta_gam):
        f_beta_dsig = - self.k * self.b * delta_gam / self.sig_y * I_ax
        f_beta_dbeta = (1.0 + self.k * self.b / self.sig_y * delta_gam + self.k * delta_gam) * I_ax
        f_beta_dtheta = self.k * self.b * delta_gam / self.sig_y * I_ax
        f_beta_dgamma = - self.k * self.b / self.sig_y * eta + self.k * (beta)
        matrices = (f_beta_dsig, f_beta_dbeta, f_beta_dtheta, np.matrix(f_beta_dgamma).transpose())
        return np.hstack(matrices)

    def calc_f_theta(self, eta, theta, a, delta_gam):
        theta_bar = np.sqrt(3 / 2) * self.calc_stress_norm(theta)
        if theta_bar == 0.0:
            return theta - self.theta - a * self.C * delta_gam / self.sig_y * eta
        return theta - self.theta - a * self.C * delta_gam / self.sig_y * eta + self.C * delta_gam * np.sqrt(a / theta_bar) * theta

    def calc_j_f_theta(self, eta, theta, a, delta_gam, hardening_flag):
        theta_bar = np.sqrt(3 / 2) * self.calc_stress_norm(theta)
        s = 1 / (1 + self.k * delta_gam)
        if hardening_flag:
            a_prime = - self.k * s**2 * (self.R + self.k * self.Rsat * delta_gam) + s * self.k * self.Rsat
        else:
            a_prime = 0.0
        f_theta_dsig = - a * self.C * delta_gam / self.sig_y * I_ax
        f_theta_dbeta = a * self.C * delta_gam / self.sig_y * I_ax
        if theta_bar == 0.0:
            f_theta_dtheta =  (1 + a * self.C * delta_gam / self.sig_y) * I_ax
            f_theta_dgamma = (
                - a * self.C / self.sig_y -
                self.C * delta_gam / self.sig_y * a_prime
            ) * eta
        else:
            n_bar_theta = self.Q @ theta / (theta_bar / np.sqrt(3 / 2))
            f_theta_dtheta =  (
                1 + a * self.C * delta_gam / self.sig_y +
                self.C * delta_gam * np.sqrt(a / theta_bar)) * I_ax - (
                    np.sqrt(3 / 2) * self.C * delta_gam * np.sqrt(a / theta_bar) / (2 * theta_bar) * np.outer(n_bar_theta, theta)
                )
            f_theta_dgamma = (
                - a * self.C / self.sig_y * eta -
                self.C * delta_gam / self.sig_y * a_prime * eta +
                self.C * np.sqrt(a / theta_bar) * theta +
                self.C * delta_gam * np.sqrt(1.0 / (theta_bar * a)) * a_prime / 2 * theta
            )
        matrices = (f_theta_dsig, f_theta_dbeta, f_theta_dtheta, np.matrix(f_theta_dgamma).transpose())
        return np.hstack(matrices)
    
    def calc_f_theta_dtheta(self, theta, a, delta_gam):
        theta_bar = np.sqrt(3 / 2) * self.calc_stress_norm(theta)
        n_bar_theta = self.Q @ theta / (theta_bar / np.sqrt(3 / 2))
        f_theta_dtheta =  (
            1 + a * self.C * delta_gam / self.sig_y +
            self.C * delta_gam * np.sqrt(a / theta_bar)) * I_ax - (
                np.sqrt(3 / 2) * self.C * delta_gam * np.sqrt(a) / (2 * theta_bar * np.sqrt(theta_bar)) * np.outer(n_bar_theta, theta)
            )
        return f_theta_dtheta

    def calc_f_theta_dgamma(self, eta, theta, a, a_prime, delta_gam):
        theta_bar = np.sqrt(3 / 2) * self.calc_stress_norm(theta)
        f_theta_dgamma = (
            - a * self.C / self.sig_y * eta -
            self.C * delta_gam / self.sig_y * a_prime * eta +
            self.C * np.sqrt(a / theta_bar) * theta +
            self.C * delta_gam * np.sqrt(1 / theta_bar / a) * a_prime / 2 * theta
        )
        return f_theta_dgamma

    def calc_jacobian(self, sig_d, sig_d_tri, eta, beta, theta, a, delta_gam, hardening_flag):
        j_f_f = self.calc_j_f_f(eta)
        j_f_ep = self.calc_j_f_ep(sig_d, sig_d_tri, eta, delta_gam)
        j_f_beta = self.calc_j_f_beta(eta, beta, delta_gam)
        j_f_theta = self.calc_j_f_theta(eta, theta, a, delta_gam, hardening_flag)
        matrices = (j_f_f, j_f_ep, j_f_beta, j_f_theta)
        return np.vstack(matrices)

    def calc_f_vector(self, sig_d, sig_d_tri, beta, theta, a, delta_gam):
        eta = sig_d - beta - theta
        f_f = self.calc_f_f(eta)
        f_ep = self.calc_f_ep(sig_d, sig_d_tri, eta, delta_gam)
        f_beta = self.calc_f_beta(eta, beta, delta_gam)
        f_theta = self.calc_f_theta(eta, theta, a, delta_gam)
        vectors = ([f_f], f_ep, f_beta, f_theta)
        return np.hstack(vectors)

    def divide_delta_vector(self, delta_vector):
        delta_sig, delta_beta, delta_theta, delta_gam_l = np.split(delta_vector, (4, 8, 12))
        delta_gam = delta_gam_l[0]
        return delta_sig, delta_beta, delta_theta, delta_gam

    def return_mapping(self, sig_d, sig_d_tri):
        delta_vector = np.zeros(13)
        delta_gam_i = 0.0
        eta_tri = sig_d_tri - self.theta - self.beta
        f_tri = self.calc_f_f(eta_tri)
        hardening_flag = True
        if f_tri > 0.0:
            logger.debug("Plastic behavior")
            sig_d_i = sig_d
            beta_i = self.beta
            theta_i = self.theta
            a_i = self.B + self.R - self.sig_y
            eta_i = sig_d_i - beta_i - theta_i
            f_vector = self.calc_f_vector(sig_d_i, sig_d_tri, beta_i, theta_i, a_i, delta_gam_i)
            jacobian = self.calc_jacobian(sig_d_i, sig_d_tri, eta_i, beta_i, theta_i, a_i, delta_gam_i, hardening_flag)
            for inew in range(self.RM_I):
                logger.debug(f"Newton iteration {inew+1}")
                d_delta_vector = np.linalg.solve(jacobian, f_vector)
                delta_vector -= np.array(d_delta_vector).flatten()
                delta_sig, delta_beta, delta_theta, delta_gam_i = self.divide_delta_vector(delta_vector)
                sig_d_i = sig_d + delta_sig
                beta_i = self.beta + delta_beta
                theta_i = self.theta + delta_theta
                xi_n = beta_i - self.q
                g_stag = self.calc_g_stag(xi_n, self.r)
                g_stag_flow = (self.Q @ xi_n) @ delta_beta
                if g_stag > -self.TOL and g_stag_flow > -self.TOL:
                    hardening_flag = True
                    R_i = 1 / (1 + self.k * delta_gam_i) * (self.R + self.k * self.Rsat * delta_gam_i)
                    a_i = self.B + R_i - self.sig_y
                else:
                    hardening_flag = False
                    a_i = self.B + self.R - self.sig_y
                eta_i = sig_d_i - beta_i - theta_i
                f_vector = self.calc_f_vector(sig_d_i, sig_d_tri, beta_i, theta_i, a_i, delta_gam_i)
                jacobian = self.calc_jacobian(sig_d_i, sig_d_tri, eta_i, beta_i, theta_i, a_i, delta_gam_i, hardening_flag)
                if np.linalg.norm(f_vector) < self.TOL:
                    if delta_gam_i < 0.0:
                        raise ValueError("Delta gamma is negative value.")
                    logger.info(f"Return map converged itr.{inew+1}")
                    logger.debug(f"Delta gamma: {delta_gam_i}")
                    break
            else:
                logger.warning("Return mapping isn't converged.")
                raise NotConvergedError
        else:
            logger.debug("Elastic behavior")
        return delta_vector, hardening_flag

    def calc_Dep(self, sig_d, delta_gam, beta, theta, hardening_flag):
        if delta_gam == 0.0:
            return self.De
        eta = sig_d - beta - theta
        theta_bar = np.sqrt(3 / 2) * self.calc_stress_norm(theta)
        g_eta, m = self.calc_g_flow(eta)
        if hardening_flag:
            a = self.B + self.R_i - self.sig_y
        else:
            a = self.B + self.R - self.sig_y
        factor = self.calc_De_factor(self.Ea, self.elastic.E, self.psi, self.eff_eps_p + delta_gam)
        D_n_n_D = factor**2 * self.elastic.De @ (np.outer(m, m) @ self.elastic.De)
        n_D_n = factor * m @ (self.elastic.De @ m)
        S = (self.C * a + self.k * self.b) / self.sig_y * eta - (self.C * np.sqrt(a / theta_bar) * theta + self.k * beta)
        n_s = m @ S
        return factor * self.elastic.De - D_n_n_D / (n_D_n + n_s)

    def integrate_stress(self, eps, del_eps):
        eps_tri = eps + del_eps
        sig_i = self.De @ (eps - self.eps_p)
        sig_d_i = Id_s_ax @ sig_i
        eps_e_tri = eps_tri - self.eps_p
        sig_tri = self.De @ eps_e_tri
        sig_d_tri = Id_s_ax @ sig_tri
        sig_v = sig_tri - sig_d_tri
        delta_vector, hardening_flag = self.return_mapping(sig_d_i, sig_d_tri)
        delta_sig, delta_beta, delta_theta, delta_gam = self.divide_delta_vector(delta_vector)
        if delta_gam == 0.0:
            sig = sig_tri
            return sig, self.De
        sig_d = sig_d_i + delta_sig
        eta = sig_d - (self.beta + delta_beta) - (self.theta + delta_theta)
        g, n_s_f = self.calc_g_flow(eta)
        self.update_i(delta_beta, delta_theta, delta_gam, n_s_f)
        sig = sig_d + sig_v
        beta = self.beta + delta_beta
        theta = self.theta + delta_theta

        Dep = self.calc_Dep(sig_d, delta_gam, beta, theta, hardening_flag)
        return sig, Dep
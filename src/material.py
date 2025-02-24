import numpy as np
from src.util import I, Is, Id, Id_s, IxI

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
    TOL = 1.0e-06
    RM_I = 10

    def __init__(self, elastic: Elastic, sig_y: float):
        self.elastic = elastic
        self.sig_y = sig_y
        self.eps = np.zeros(6)
        self.eps_p = np.zeros(6)
        self.eps_p_i = np.zeros(6)
        self.eff_eps_p = 0.0
        self.eff_eps_p_i = 0.0

    @staticmethod
    def calc_stress_norm(sig):
        r = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
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
        print("-"*80)
        del_gam = 0.0
        f_ip1 = self.calc_f_ip1(sig_d_tri, del_gam)
        f_ip1_prime = self.calc_f_ip1_prime(sig_d_tri, del_gam)
        if f_ip1 < 0.0:
            print("Plastic behavior")
            for inew in range(self.RM_I):
                print(f"Newton iteration {inew+1}")
                d_del_gam = f_ip1 / f_ip1_prime
                del_gam -= d_del_gam
                f_ip1 = self.calc_f_ip1(sig_d_tri, del_gam)
                f_ip1_prime = self.calc_f_ip1_prime(sig_d_tri, del_gam)
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
        q_tri, n_bar = self.calc_tri(sig_d_tri, del_gam)
        return q_tri, del_gam, n_bar

    def integrate_stress(self, eps, del_eps):
        eps_tri = eps + del_eps
        eps_e_tri = eps_tri - self.eps_p
        sig_tri = self.elastic.De @ eps_e_tri
        sig_d_tri = Id_s @ sig_tri

        q_tri, del_gam, n_bar = self.return_mapping(sig_d_tri)
        print(f"F: {self.calc_f_ip1(sig_d_tri, del_gam)}")
        self.update_i(del_gam, n_bar)
        eps_e_i = eps_tri - self.eps_p_i
        sig_e = self.elastic.De @ eps_e_i
        sig = sig_tri - 2 * self.elastic.G * del_gam * n_bar * np.sqrt(3 / 2)
        Dep = self.calc_Dep(sig_d_tri, del_gam)
        return sig, Dep


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

    def update_i(self, del_gam, n_bar):
        self.r_i = self.r + self.h * del_gam
        delta_eps_p = np.sqrt(3 / 2) * del_gam * n_bar * np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
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
             6 * self.elastic.G**2 * del_gam / q_tri * Id +
             6 * self.elastic.G**2 * (del_gam / q_tri - 1 / f_ip1_prime) * np.outer(n_bar, n_bar)
         )


class Linear_kinematic(Material_expression_base):
    def __init__(self, elastic: Elastic, sig_y: float, h: float):
        super().__init__(elastic, sig_y)
        self.h = h

    def initialize(self):
        self.alpha = np.zeros(6)
        self.alpha_i = np.zeros(6)
        self.eps_p = np.zeros(6)
        self.eps_p_i = np.zeros(6)
        self.eff_eps_p = 0.0
        self.eff_eps_p_i = 0.0

    def update_i(self, del_gam, n_bar):
        delta_eps_p = np.sqrt(3 / 2) * del_gam * n_bar * np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
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
        self.eps_p = np.zeros(6)
        self.eps_p_i = np.zeros(6)
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
             6 * self.elastic.G**2 * del_gam / q_tri * Id +
             6 * self.elastic.G**2 * (del_gam / q_tri - 1 / f_ip1_prime) * np.outer(n_bar, n_bar)
         )


class Voce_isotropic_n(Material_expression_base):
    def __init__(self, elastic: Elastic, sig_y: float, Qs: list[float], bs: list[float]):
        super().__init__(elastic, sig_y)
        self.Qs = Qs
        self.bs = bs

    @property
    def number(self):
        return len(self.Qs)

    @property
    def yield_stress(self):
        return self.sig_y + self.r

    @property
    def r(self):
        return sum(self.rs)

    @property
    def r_i(self):
        return sum(self.r_is)

    def initialize(self):
        self.rs = [0.0] * self.number
        self.r_is = [0.0] * self.number
        self.eps_p = np.zeros(6)
        self.eps_p_i = np.zeros(6)
        self.eff_eps_p = 0.0
        self.eff_eps_p_i = 0.0

    def update_i(self, del_gam, n_bar):
        for n in range(self.number):
            theta_n = 1 / (1 + self.bs[n] * del_gam)
            self.r_is[n] = theta_n * (self.rs[n] + self.bs[n] * self.Qs[n] * del_gam)
        delta_eps_p = np.sqrt(3 / 2) * del_gam * n_bar * np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        self.eps_p_i = self.eps_p + delta_eps_p
        self.eff_eps_p_i = self.eff_eps_p + del_gam

    def update(self):
        for n in range(self.number):
            self.rs[n] = self.r_is[n]
        self.eps_p = self.eps_p_i
        self.eff_eps_p = self.eff_eps_p_i

    def calc_tri(self, sig_d, del_gam):
        norm = self.calc_stress_norm(sig_d)
        return np.sqrt(3 / 2) * norm, sig_d / norm

    def calc_f_ip1(self, sig_d: np.ndarray, del_gam: float):
        q_tri, n_bar = self.calc_tri(sig_d, del_gam)
        G = 0.0
        for n in range(self.number):
            theta_n = 1 / (1 + self.bs[n] * del_gam)
            G += theta_n * (self.rs[n] + self.bs[n] * self.Qs[n] * del_gam)
        return self.sig_y + G + 3 * self.elastic.G * del_gam - q_tri

    def calc_f_ip1_prime(self, sig_d: np.ndarray, del_gam: float):
        G_prime = 0.0
        for n in range(self.number):
            theta_n = 1 / (1 + self.bs[n] * del_gam)
            G_prime += -self.bs[n] * theta_n**2 * (self.rs[n] + self.bs[n] * self.Qs[n] * del_gam) + theta_n * self.bs[n] * self.Qs[n]
        return 3 * self.elastic.G + G_prime

    def calc_Dep(self, sig_d: np.ndarray, del_gam: float):
         q_tri, n_bar = self.calc_tri(sig_d, del_gam)
         if del_gam == 0.0:
             return self.elastic.De
         f_ip1_prime = self.calc_f_ip1_prime(sig_d, del_gam)
         return (
             self.elastic.De -
             6 * self.elastic.G**2 * del_gam / q_tri * Id +
             6 * self.elastic.G**2 * (del_gam / q_tri - 1 / f_ip1_prime) * np.outer(n_bar, n_bar)
         )


class AF_kinematic(Material_expression_base):
    def __init__(self, elastic: Elastic, sig_y: float, C: float, k: float):
        super().__init__(elastic, sig_y)
        self.C = C
        self.k = k

    def initialize(self):
        self.alpha = np.zeros(6)
        self.alpha_i = np.zeros(6)
        self.eps_p = np.zeros(6)
        self.eps_p_i = np.zeros(6)
        self.eff_eps_p = 0.0
        self.eff_eps_p_i = 0.0

    def update_i(self, del_gam, n_bar):
        delta_eps_p = np.sqrt(3 / 2) * del_gam * n_bar * np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
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
        N4d = I - np.outer(n_bar, n_bar)
        N4d_alpha = N4d @ self.alpha
        
        return (
            self.elastic.De -
            6 * self.elastic.G**2 * del_gam / q_tri * Id +
            6 * self.elastic.G**2 * (del_gam / q_tri - 1 / f_ip1_prime) * np.outer(n_bar, n_bar) -
            3 * np.sqrt(6) * self.elastic.G**2 * self.k * theta**2 * del_gam / (q_tri * f_ip1_prime) * np.outer(N4d_alpha, n_bar)
        )


class AF_kinematic_n(Material_expression_base):
    def __init__(self, elastic: Elastic, sig_y: float, Cs: list[float], ks: list[float]):
        super().__init__(elastic, sig_y)
        self.Cs = Cs
        self.ks = ks

    @property
    def number(self):
        return len(self.Cs)

    @property
    def alpha(self):
        return sum(self.alphas)

    @property
    def alpha_i(self):
        return sum(self.alpha_is)

    def initialize(self):
        self.alphas = [np.zeros(6)]*self.number
        self.alpha_is = [np.zeros(6)]*self.number
        self.eps_p = np.zeros(6)
        self.eps_p_i = np.zeros(6)
        self.eff_eps_p = 0.0
        self.eff_eps_p_i = 0.0

    def update_i(self, del_gam, n_bar):
        delta_eps_p = np.sqrt(3 / 2) * del_gam * n_bar * np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        for n in range(self.number):
            theta_n = 1 / (1 + self.ks[n] * del_gam)
            self.alpha_is[n] = theta_n * (self.alphas[n] + np.sqrt(2 / 3) * self.Cs[n] * del_gam * n_bar)
        self.eps_p_i = self.eps_p + delta_eps_p
        self.eff_eps_p_i = self.eff_eps_p + del_gam

    def update(self):
        self.eps_p = self.eps_p_i
        for n in range(self.number):
            self.alphas[n] = self.alpha_is[n]
        self.eff_eps_p = self.eff_eps_p_i

    def calc_tri(self, sig_d: np.ndarray, del_gam: float):
        eta = sig_d - self.alpha
        norm = self.calc_stress_norm(eta)
        return np.sqrt(3 / 2) * norm, eta / norm

    def calc_f_ip1(self, sig_d: np.ndarray, del_gam: float):
        q_tri, n_bar = self.calc_tri(sig_d, del_gam)
        E = 0.0
        for n in range(self.number):
            theta_n = 1 / (1 + self.ks[n] * del_gam)
            E += self.Cs[n] * theta_n * del_gam
        return self.sig_y + 3 * self.elastic.G * del_gam + E - q_tri

    def calc_f_ip1_prime(self, sig_d: np.ndarray, del_gam: float):
        E_prime = 0.0
        for n in range(self.number):
            theta_n = 1 / (1 + self.ks[n] * del_gam)
            E_prime += self.Cs[n] * (theta_n - self.ks[n] * theta_n**2 * del_gam)
        return 3 * self.elastic.G + E_prime

    def calc_Dep(self, sig_d: np.ndarray, del_gam: float):
        q_tri, n_bar = self.calc_tri(sig_d, del_gam)
        if del_gam == 0.0:
            return self.elastic.De
        f_ip1_prime = self.calc_f_ip1_prime(sig_d, del_gam)
        N4d = I - np.outer(n_bar, n_bar)
        N4d_alpha = np.zeros(6)
        for n in range(self.number):
            theta_n = 1 / (1 + self.ks[n] * del_gam)
            N4d_alpha += self.ks[n] * theta_n**2 * N4d @ self.alphas[n]
        
        return (
            self.elastic.De -
            6 * self.elastic.G**2 * del_gam / q_tri * Id +
            6 * self.elastic.G**2 * (del_gam / q_tri - 1 / f_ip1_prime) * np.outer(n_bar, n_bar) -
            3 * np.sqrt(6) * self.elastic.G**2 * del_gam / (q_tri * f_ip1_prime) * np.outer(N4d_alpha, n_bar)
        )


class Chaboche(Material_expression_base):
    def __init__(self, elastic: Elastic, sig_y: float, C: float, k: float, Q: float, b: float):
        super().__init__(elastic, sig_y)
        self.C = C
        self.k = k
        self.Q = Q
        self.b = b

    @property
    def yield_stress(self):
        return self.sig_y + self.r

    def initialize(self):
        self.r = 0.0
        self.r_i = 0.0
        self.alpha = np.zeros(6)
        self.alpha_i = np.zeros(6)
        self.eps_p = np.zeros(6)
        self.eps_p_i = np.zeros(6)
        self.eff_eps_p = 0.0
        self.eff_eps_p_i = 0.0

    def update_i(self, del_gam, n_bar):
        theta_i = 1 / (1 + self.b * del_gam)
        self.r_i = theta_i * (self.r + self.b * self.Q * del_gam)
        delta_eps_p = np.sqrt(3 / 2) * del_gam * n_bar * np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        theta_k = 1 / (1 + self.k * del_gam)
        self.alpha_i = theta_k * (self.alpha + np.sqrt(2 / 3) * self.C * del_gam * n_bar)
        self.eps_p_i = self.eps_p + delta_eps_p
        self.eff_eps_p_i = self.eff_eps_p + del_gam

    def update(self):
        self.r = self.r_i
        self.eps_p = self.eps_p_i
        self.alpha = self.alpha_i
        self.eff_eps_p = self.eff_eps_p_i

    def calc_tri(self, sig_d: np.ndarray, del_gam: float):
        eta = sig_d - self.alpha
        norm = self.calc_stress_norm(eta)
        return np.sqrt(3 / 2) * norm, eta / norm

    def calc_f_ip1(self, sig_d: np.ndarray, del_gam: float):
        q_tri, n_bar = self.calc_tri(sig_d, del_gam)
        theta_k = 1 / (1 + self.k * del_gam)
        theta_i = 1 / (1 + self.b * del_gam)
        return self.sig_y + 3 * self.elastic.G * del_gam + self.C * theta_k * del_gam + theta_i * (self.r + self.b * self.Q * del_gam) - q_tri

    def calc_f_ip1_prime(self, sig_d: np.ndarray, del_gam: float):
        theta_i = 1 / (1 + self.b * del_gam)
        theta_k = 1 / (1 + self.k * del_gam)
        return (
            3 * self.elastic.G +
            self.C * theta_k -
            self.k * self.C * theta_k**2 * del_gam -
            self.b * theta_i**2 * (self.r + self.b * self.Q * del_gam) +
            self.b * self.Q * theta_i
        )

    def calc_Dep(self, sig_d: np.ndarray, del_gam: float):
        q_tri, n_bar = self.calc_tri(sig_d, del_gam)
        if del_gam == 0.0:
            return self.elastic.De
        f_ip1_prime = self.calc_f_ip1_prime(sig_d, del_gam)
        theta_k = 1 / (1 + self.k * del_gam)
        N4d = I - np.outer(n_bar, n_bar)
        N4d_alpha = N4d @ self.alpha
        
        return (
            self.elastic.De -
            6 * self.elastic.G**2 * del_gam / q_tri * Id +
            6 * self.elastic.G**2 * (del_gam / q_tri - 1 / f_ip1_prime) * np.outer(n_bar, n_bar) -
            3 * np.sqrt(6) * self.elastic.G**2 * self.k * theta_k**2 * del_gam / (q_tri * f_ip1_prime) * np.outer(N4d_alpha, n_bar)
        )


class Chaboche_n(Material_expression_base):
    def __init__(self, elastic: Elastic, sig_y: float, Cs: list[float], ks: list[float], Qs: list[float], bs: list[float]):
        super().__init__(elastic, sig_y)
        self.Cs = Cs
        self.ks = ks
        self.Qs = Qs
        self.bs = bs

    @property
    def yield_stress(self):
        return self.sig_y + self.r

    @property
    def number_k(self):
        return len(self.Cs)

    @property
    def number_i(self):
        return len(self.Qs)

    @property
    def alpha(self):
        return sum(self.alphas)

    @property
    def alpha_i(self):
        return sum(self.alpha_is)

    @property
    def r(self):
        return sum(self.rs)

    @property
    def r_i(self):
        return sum(self.r_is)


    def initialize(self):
        self.alphas = [np.zeros(6)]*self.number_k
        self.alpha_is = [np.zeros(6)]*self.number_k
        self.rs = [0.0] * self.number_i
        self.r_is = [0.0] * self.number_i
        self.eps_p = np.zeros(6)
        self.eps_p_i = np.zeros(6)
        self.eff_eps_p = 0.0
        self.eff_eps_p_i = 0.0

    def update_i(self, del_gam, n_bar):
        delta_eps_p = np.sqrt(3 / 2) * del_gam * n_bar * np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        for n_k in range(self.number_k):
            theta_n_k = 1 / (1 + self.ks[n_k] * del_gam)
            self.alpha_is[n_k] = theta_n_k * (self.alphas[n_k] + np.sqrt(2 / 3) * self.Cs[n_k] * del_gam * n_bar)
        for n_i in range(self.number_i):
            theta_n_i = 1 / (1 + self.bs[n_i] * del_gam)
            self.r_is[n_i] = theta_n_i * (self.rs[n_i] + self.bs[n_i] * self.Qs[n_i] * del_gam)
        self.eps_p_i = self.eps_p + delta_eps_p
        self.eff_eps_p_i = self.eff_eps_p + del_gam

    def update(self):
        self.eps_p = self.eps_p_i
        for n_k in range(self.number_k):
            self.alphas[n_k] = self.alpha_is[n_k]
        for n_i in range(self.number_i):
            self.rs[n_i] = self.r_is[n_i]
        self.eff_eps_p = self.eff_eps_p_i

    def calc_tri(self, sig_d: np.ndarray, del_gam: float):
        eta = sig_d - self.alpha
        norm = self.calc_stress_norm(eta)
        return np.sqrt(3 / 2) * norm, eta / norm

    def calc_f_ip1(self, sig_d: np.ndarray, del_gam: float):
        q_tri, n_bar = self.calc_tri(sig_d, del_gam)
        E = 0.0
        for n_k in range(self.number_k):
            theta_n_k = 1 / (1 + self.ks[n_k] * del_gam)
            E += self.Cs[n_k] * theta_n_k * del_gam
        G = 0.0
        for n_i in range(self.number_i):
            theta_n_i = 1 / (1 + self.bs[n_i] * del_gam)
            G += theta_n_i * (self.rs[n_i] + self.bs[n_i] * self.Qs[n_i] * del_gam)
        return self.sig_y + 3 * self.elastic.G * del_gam + E + G - q_tri

    def calc_f_ip1_prime(self, sig_d: np.ndarray, del_gam: float):
        q_tri, n_bar = self.calc_tri(sig_d, del_gam)
        E_prime = 0.0
        for n_k in range(self.number_k):
            theta_n_k = 1 / (1 + self.ks[n_k] * del_gam)
            E_prime += self.Cs[n_k] * (theta_n_k - self.ks[n_k] * theta_n_k**2 * del_gam)
        G_prime = 0.0
        for n_i in range(self.number_i):
            theta_n_i = 1 / (1 + self.bs[n_i] * del_gam)
            G_prime += -self.bs[n_i] * theta_n_i**2 * (self.rs[n_i] + self.bs[n_i] * self.Qs[n_i] * del_gam) + theta_n_i * self.bs[n_i] * self.Qs[n_i]
        return 3 * self.elastic.G + E_prime + G_prime

    def calc_Dep(self, sig_d: np.ndarray, del_gam: float):
        q_tri, n_bar = self.calc_tri(sig_d, del_gam)
        if del_gam == 0.0:
            return self.elastic.De
        f_ip1_prime = self.calc_f_ip1_prime(sig_d, del_gam)
        N4d = I - np.outer(n_bar, n_bar)
        N4d_alpha = np.zeros(6)
        for n_k in range(self.number_k):
            theta_n_k = 1 / (1 + self.ks[n_k] * del_gam)
            N4d_alpha += self.ks[n_k] * theta_n_k**2 * N4d @ self.alphas[n_k]
        
        return (
            self.elastic.De -
            6 * self.elastic.G**2 * del_gam / q_tri * Id +
            6 * self.elastic.G**2 * (del_gam / q_tri - 1 / f_ip1_prime) * np.outer(n_bar, n_bar) -
            3 * np.sqrt(6) * self.elastic.G**2 * del_gam / (q_tri * f_ip1_prime) * np.outer(N4d_alpha, n_bar)
        )


class Yoshida_uemori:
    TOL = 1.0e-06
    RM_I = 10

    def __init__(self, elastic: Elastic, sig_y: float, B, C, Rsat, k, b, h):
        self.elastic = elastic
        self.sig_y = sig_y
        self.B = B
        self.C = C
        self.Rsat = Rsat
        self.k = k
        self.b = b
        self.h = h
        self.eps_p = np.zeros(6)
        self.eps_p_i = np.zeros(6)
        self.eff_eps_p = 0.0
        self.eff_eps_p_i = 0.0
        self.theta = np.zeros(6)
        self.theta_i = np.zeros(6)
        self.beta = np.zeros(6)
        self.beta_i = np.zeros(6)
        self.R = 0.0
        self.R_i = 0.0
        self.r = 0.0
        self.r_i = 0.0
        self.q = np.zeros(6)
        self.q_i = np.zeros(6)

    @staticmethod
    def calc_stress_norm(sig):
        r = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        sig_r = sig * r
        return np.sqrt(sig @ sig_r)

    @property
    def yield_stress(self):
        return self.sig_y

    @property
    def alpha(self):
        return self.theta + self.beta

    @property
    def a(self):
        return self.B + self.R + self.sig_y

    def initialize(self):
        self.eps_p = np.zeros(6)
        self.eps_p_i = np.zeros(6)
        self.eff_eps_p = 0.0
        self.eff_eps_p_i = 0.0
        self.theta = np.zeros(6)
        self.theta_i = np.zeros(6)
        self.beta = np.zeros(6)
        self.beta_i = np.zeros(6)
        self.r = 0.0
        self.r_i = 0.0
        self.q = np.zeros(6)
        self.q_i = np.zeros(6)


    def update_i(self, delta_beta, delta_theta, delta_gam, n_bar):
        self.beta_i += delta_beta
        self.theta_i += delta_theta
        self.eps_p_i += np.sqrt(3 / 2) * n_bar * delta_gam
        self.eff_eps_p_i += delta_gam
        self.R_i += self.k * (self.Rsat - self.R) * delta_gam

    def update(self):
        self.eps_p = self.eps_p_i
        self.eff_eps_p = self.eff_eps_p_i
        self.beta = self.beta_i
        self.theta = self.theta_i
        self.R = self.R_i

    def calc_g(self, sig_d):
        norm = self.calc_stress_norm(sig_d)
        n_bar = sig_d / norm
        return np.sqrt(3 / 2) * norm, np.sqrt(3 / 2) * n_bar

    def calc_f_f(self, eta):
        g_eta, n_s = self.calc_g(eta)
        return g_eta - self.sig_y

    def calc_j_f_f(self, eta):
        g_eta, n_s = self.calc_g(eta)
        f_f_dsig = n_s
        f_f_dbeta = - n_s
        f_f_dtheta = - n_s
        f_f_dgamma = [0.0]
        return np.hstack(f_f_dsig, f_f_dbeta, f_f_dtheta, f_f_dgamma)

    def calc_f_ep(self, sig, sig_tri, n_s, delta_gam):
        return (self.elastic.De_inv @ (sig - sig_tri)) + delta_gam * n_s

    def calc_j_f_ep(self, eta, delta_gam):
        g_eta, n_s = self.calc_g(eta)
        dn_dsig = 3 /(2 * g_eta) * (I - np.outer(n_s / np.sqrt(3 / 2), n_s / np.sqrt(3 / 2)))
        f_ep_dsig = self.elastic.De_inv + delta_gam * dn_dsig
        f_ep_dbeta = - delta_gam * dn_dsig
        f_ep_dtheta = - delta_gam * dn_dsig
        f_ep_dgamma = n_s
        matrices = (f_ep_dsig, f_ep_dbeta, f_ep_dtheta, np.matrix(f_ep_dgamma).transpose())
        return np.hstack(matrices)

    def calc_f_beta(self, eta, delta_beta, delta_gam):
        return delta_beta - (self.k * self.b / self.sig_y * eta - self.k * (self.beta + delta_beta)) * delta_gam

    def calc_j_f_beta(self, eta, delta_beta, delta_gam):
        f_beta_dsig = - self.k * self.b * delta_gam / self.sig_y * I
        f_beta_dbeta = (1.0 + self.k * self.b / self.sig_y * delta_gam + self.k * delta_gam) * I
        f_beta_dtheta = self.k * self.b * delta_gam / self.sig_y * I
        f_beta_dgamma = - self.k * self.b / self.sig_y * eta + self.k * (self.beta + delta_beta)
        matrices = (f_beta_dsig, f_beta_dbeta, f_beta_dtheta, np.matrix(f_beta_dgamma).transpose())
        return np.hstack(matrices)

    def calc_f_theta(self, eta, theta, a, delta_gam, n_s):
        theta_bar = np.sqrt(3 / 2) * self.calc_stress_norm(theta)
        return theta - self.theta - a * self.C * delta_gam / self.sig_y * eta + self.C * delta_gam * np.sqrt(a / theta_bar) * theta

    def calc_f_theta_dsig(self, a, delta_gam):
        return - a * self.C * delta_gam / self.sig_y * I

    def calc_f_theta_dbeta(self, a, delta_gam):
        return a * self.C * delta_gam / self.sig_y * I


    def calc_f_vector(self, sig, sig_tri, beta, theta, delta_gam, n_s):
        eta = Id_s @ sig - beta - theta
        f_f = self.calc_f_f(eta)
        f_ep = self.calc_f_ep(sig, sig_tri, n_s, delta_gam)
        f_beta = self.calc_f_beta(beta, n_s, delta_gam)
        f_theta = self.calc_f_theta(theta, a, delta_gam) 

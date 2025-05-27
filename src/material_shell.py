import numpy as np

Id_s = 1 / 3 * np.array([[2.0, -1.0, 0.0], [-1.0, 2.0, 0.0], [0.0, 0.0, 6.0]])


class Elastic_shell:
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
        E = self.E
        n = self.n
        return E / (1 - n**2) * np.array([[1, n, 0], [n, 1, 0], [0, 0, (1 - n) / 2]])

    @property
    def De_inv(self):
        return np.linalg.inv(self.De)


class Material_expression_base_shell:
    TOL = 1.0e-06
    RM_I = 10

    def __init__(self, elastic: Elastic_shell, sig_y: float):
        self.elastic = elastic
        self.sig_y = sig_y
        self.eps = np.zeros(3)
        self.eps_p = np.zeros(3)
        self.eps_p_i = np.zeros(3)
        self.eff_eps_p = 0.0
        self.eff_eps_p_i = 0.0

    @staticmethod
    def calc_stress_norm(sig):
        r = np.array([1.0, 1.0, 2.0])
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

    def return_mapping(self, sig_tri):
        print("-" * 80)
        del_gam = 0.0
        f_ip1 = self.calc_f_ip1(sig_tri, del_gam)
        f_ip1_prime = self.calc_f_ip1_prime(sig_tri, del_gam)
        if f_ip1 < 0.0:
            print("Plastic behavior")
            for inew in range(self.RM_I):
                print(f"Newton iteration {inew+1}")
                d_del_gam = f_ip1 / f_ip1_prime
                del_gam -= d_del_gam
                f_ip1 = self.calc_f_ip1(sig_tri, del_gam)
                f_ip1_prime = self.calc_f_ip1_prime(sig_tri, del_gam)
                print(f_ip1)
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
        q_tri, n_bar = self.calc_tri(sig_tri, del_gam)
        return q_tri, del_gam, n_bar

    def integrate_stress(self, eps, del_eps):
        eps_tri = eps + del_eps
        eps_e_tri = eps_tri - self.eps_p
        sig_tri = self.elastic.De @ eps_e_tri

        q_tri, del_gam, n_bar = self.return_mapping(sig_tri)
        print(f"q_tri: {q_tri}")
        print(f"del_gam: {del_gam}")
        print(f"n_bar: {n_bar}")
        print(f"F: {self.calc_f_ip1(sig_tri, del_gam)}")
        self.update_i(q_tri, del_gam, n_bar)
        eps_e_i = eps_tri - self.eps_p_i
        sig_e = self.elastic.De @ eps_e_i
        sig = sig_tri - 2 * self.elastic.G * del_gam * n_bar * np.sqrt(3 / 2)
        Dep = self.calc_Dep(sig_tri, del_gam)
        print(f"sig: {sig}")
        print(f"DepSig: {Dep @ eps}")
        return sig, Dep


class Linear_isotropic_shell(Material_expression_base_shell):
    def __init__(self, elastic: Elastic_shell, sig_y: float, h: float):
        self.elastic = elastic
        self.sig_y = sig_y
        self.h = h
        self.r = 0.0
        self.r_i = 0.0
        self.eps_p = np.zeros(3)
        self.eps_p_i = np.zeros(3)
        self.eff_eps_p = 0.0
        self.eff_eps_p_i = 0.0

    @property
    def yield_stress(self):
        return self.sig_y + self.r

    def initialize(self):
        self.r = 0.0
        self.r_i = 0.0

    def update_i(self, q_tri, del_gam, n_bar):
        self.eps_p_i = self.eps_p + del_gam * n_bar * np.array([1.0, 1.0, 2.0])
        del_eff_eps_p = del_gam * np.sqrt(2 / 3 * q_tri)
        self.eff_eps_p_i = self.eff_eps_p + del_eff_eps_p
        self.r_i = self.r + self.h * del_eff_eps_p

    def update(self):
        self.r = self.r_i
        self.eps_p = self.eps_p_i
        self.eff_eps_p = self.eff_eps_p_i

    def calc_tri(self, sig, del_gam):
        q_tri = 3 / 2 * sig @ (Id_s @ sig)
        n_bar = Id_s @ sig
        return q_tri, n_bar

    def calc_f_ip1(self, sig_d, del_gam):
        q_tri, n_bar = self.calc_tri(sig_d, del_gam)
        # d_eff_eps_p = del_gam * n_bar
        return (
            self.sig_y + (self.r + self.h * del_gam) + 3 * self.elastic.G * del_gam
        ) ** 2 - q_tri

    def calc_f_ip1_prime(self, sig_d, del_gam):
        return 3 * self.elastic.G + self.h

    def calc_Dep(self, sig, del_gam):
        q_tri, n_bar = self.calc_tri(sig, del_gam)
        if del_gam == 0.0:
            return self.elastic.De
        f_ip1_prime = self.calc_f_ip1_prime(sig, del_gam)
        return (
            self.elastic.De
            - 6 * self.elastic.G**2 * del_gam / q_tri * Id_s
            + 6
            * self.elastic.G**2
            * (del_gam / q_tri - 1 / f_ip1_prime)
            * np.outer(n_bar, n_bar)
        )

import numpy as np
from src.material import Elastic, Linear_isotropic, AF_kinematic, Chaboche_n, Yoshida_uemori
from src.util import Id_s, Id

NTENS = 3
E = 205000.0
NU = 0.3

class Test_elastic:
    def test_De(self):
        # arange
        De_expect = np.zeros([6, 6])
        lame = NU * E / (1 + NU) / (1 - 2 * NU)
        G = E / 2 / (1 + NU)
        G2 = 2 * G
        for i_m in range(NTENS):
            De_expect[i_m, i_m] += G2
            for i_n in range(NTENS):
                De_expect[i_m, i_n] += lame
        for i_m in range(NTENS, 2 * NTENS):
            De_expect[i_m, i_m] += G

        # act
        De_act = Elastic(205000.0, 0.3).De
        # assert
        assert np.allclose(De_act, De_expect, atol=1.0e-6)


class Test_linear_iso:
    MAT = Linear_isotropic(Elastic(205000.0, 0.3), 300, 20000.0)
    
    def test_tri(self):
        # arange
        sig = np.array([0.0, 0.0, 0.0, 200.0, 0.0, 0.0])
        sig_m = 1 / 3 * (sig[0] + sig[1] + sig[2])
        q_tri_expected = np.sqrt(
            0.5 * (sig[0] - sig[1]) * (sig[0] - sig[1]) +
            0.5 * (sig[1] - sig[2]) * (sig[1] - sig[2]) +
            0.5 * (sig[2] - sig[0]) * (sig[2] - sig[0]) +
            3 * (
                sig[3] * sig[3] +
                sig[4] * sig[4] +
                sig[5] * sig[5]
            )
        )
        sig_d = np.zeros(2 * NTENS)
        for i in range(NTENS):
            sig_d[i] = sig[i] - sig_m
        for i in range(NTENS, 2 * NTENS):
            sig_d[i] = sig[i]

        n_bar_expected = np.zeros(6)
        for i in range(2 * NTENS):
            n_bar_expected[i] = sig_d[i] / (q_tri_expected / np.sqrt(3 / 2))
        
        # act
        q_tri_act, n_bar_act = self.MAT.calc_tri(sig_d)

        # assert
        assert q_tri_act == q_tri_expected
        assert np.allclose(n_bar_act, n_bar_expected, atol=1.0e-6)

    def test_Dep(self):
        # arange
        Dep_expect = np.zeros([6, 6])
        lame = NU * E / (1 + NU) / (1 - 2 * NU)
        G = E / 2 / (1 + NU)
        G2 = 2 * G
        self.MAT.initialize()

        sig = np.array([300.0, 5.0, 30.0, 10.0, 20.0, 50.0])
        sig_d = Id_s @ sig
        q_tri, n_bar = self.MAT.calc_tri(sig_d)
        del_gam = 1.0e-5
        self.MAT.update_i(del_gam, n_bar)
        self.MAT.update()
        f_ip1_prime = self.MAT.calc_f_ip1_prime(q_tri, del_gam, n_bar)
        efG2 = G2 * G * del_gam / q_tri
        efG6 = 3 * efG2
        efG3 = 0.5 * efG6
        ghard = 6 * G * G * (del_gam / q_tri - 1 / f_ip1_prime)

        for i_m in range(NTENS):
            Dep_expect[i_m, i_m] += G2 - efG6
            for i_n in range(NTENS):
                Dep_expect[i_m, i_n] += lame + efG2
        for i_m in range(NTENS, 2 * NTENS):
            Dep_expect[i_m, i_m] += G - efG3
        for i_m in range(2 * NTENS):
            for i_n in range(2 * NTENS):
                Dep_expect[i_m, i_n] += ghard * n_bar[i_m] * n_bar[i_n]

        # act
        Dep_act = self.MAT.calc_Dep(q_tri, del_gam, n_bar)
        # assert
        assert np.allclose(Dep_act, Dep_expect, atol=1.0e-6)


class Test_af_kin:
    MAT = AF_kinematic(Elastic(205000.0, 0.3), 250, 20000.0, 30.0)

    def test_Dep(self):
        # arange
        Dep_expect = np.zeros([6, 6])
        lame = NU * E / (1 + NU) / (1 - 2 * NU)
        G = E / 2 / (1 + NU)
        G2 = 2 * G
        self.MAT.initialize()

        sig = np.array([300.0, 5.0, 30.0, 10.0, 20.0, 50.0])
        sig_d = Id_s @ sig
        q_tri, n_bar = self.MAT.calc_tri(sig_d)
        del_gam = 1.0e-3
        self.MAT.update_i(del_gam, n_bar)
        self.MAT.update()
        theta = 1 / (1 + self.MAT.k * del_gam)
        f_ip1_prime = self.MAT.calc_f_ip1_prime(q_tri, del_gam, n_bar)
        efG2 = G2 * G * del_gam / q_tri
        efG6 = 3 * efG2
        efG3 = 0.5 * efG6
        ghard1 = 6 * G * G * (del_gam / q_tri - 1 / f_ip1_prime)
        ghard2 = 3 * np.sqrt(6) * G * G * self.MAT.k * theta * theta * del_gam / (q_tri * f_ip1_prime)

        n4d_alpha = np.zeros(6)
        for i_m in range(2 * NTENS):
            n4d_alpha[i_m] = self.MAT.alpha[i_m]
            for i_n in range(2 * NTENS):
                n4d_alpha[i_m] -= n_bar[i_m]*n_bar[i_n]*self.MAT.alpha[i_n]

        for i_m in range(NTENS):
            Dep_expect[i_m, i_m] = 2 * G - efG6
            for i_n in range(NTENS):
                Dep_expect[i_m, i_n] += lame + efG2
        for i_m in range(NTENS, 2 * NTENS):
            Dep_expect[i_m, i_m] = G - efG3
        for i_m in range(2 * NTENS):
            for i_n in range(2 * NTENS):
                Dep_expect[i_m, i_n] += ghard1 * n_bar[i_m] * n_bar[i_n] - ghard2 * n4d_alpha[i_m] * n_bar[i_n]

        # act
        Dep_act = self.MAT.calc_Dep(q_tri, del_gam, n_bar)
        # assert
        assert np.allclose(Dep_act, Dep_expect, atol=1.0e-6)


class Test_chaboche_n:
    MAT = Chaboche_n(Elastic(205000.0, 0.3), 250, [20000.0, 10000.0], [30.0, 15.0], [100.0, 50.0], [40.0, 10.0])

    def test_Dep(self):
        # arange
        Dep_expect = np.zeros([6, 6])
        lame = NU * E / (1 + NU) / (1 - 2 * NU)
        G = E / 2 / (1 + NU)
        G2 = 2 * G
        self.MAT.initialize()

        sig = np.array([300.0, 5.0, 30.0, 10.0, 20.0, 50.0])
        sig_d = Id_s @ sig
        q_tri, n_bar = self.MAT.calc_tri(sig_d)
        del_gam = 1.0e-3
        self.MAT.update_i(del_gam, n_bar)
        self.MAT.update()
        f_ip1_prime = self.MAT.calc_f_ip1_prime(q_tri, del_gam, n_bar)
        efG2 = G2 * G * del_gam / q_tri
        efG6 = 3 * efG2
        efG3 = 0.5 * efG6
        ghard1 = 6 * G * G * (del_gam / q_tri - 1 / f_ip1_prime)
        ghard2 = 3 * np.sqrt(6) * G * G * del_gam / (q_tri * f_ip1_prime)


        n4d = np.zeros((6, 6))
        for i_m in range(2 * NTENS):
            n4d[i_m, i_m] += 1.0
            for i_n in range(2 * NTENS):
                n4d[i_m, i_n] -= n_bar[i_m] * n_bar[i_n]

        n4d_alpha = np.zeros(6)
        for i_m in range(2 * NTENS):
            for i_n in range(2 * NTENS):
                for n in range(self.MAT.number_k):
                    theta_n = 1 / (1 + self.MAT.ks[n] * del_gam)
                    n4d_alpha[i_m] += self.MAT.ks[n] * theta_n * theta_n * n4d[i_m, i_n] * self.MAT.alphas[n][i_n]

        for i_m in range(NTENS):
            Dep_expect[i_m, i_m] += 2 * G - efG6
            for i_n in range(NTENS):
                Dep_expect[i_m, i_n] += lame + efG2
        for i_m in range(NTENS, 2 * NTENS):
            Dep_expect[i_m, i_m] += G - efG3
        for i_m in range(2 * NTENS):
            for i_n in range(2 * NTENS):
                Dep_expect[i_m, i_n] += ghard1 * n_bar[i_m] * n_bar[i_n] - ghard2 * n4d_alpha[i_m] * n_bar[i_n]

        # act
        Dep_act = self.MAT.calc_Dep(q_tri, del_gam, n_bar)
        # assert
        assert np.allclose(Dep_act, Dep_expect, atol=1.0e-6)


class Test_yoshida_uemori:
    YU = Yoshida_uemori(Elastic(205000.0, 0.3), 200.0, 160.0, 500.0, 120.0, 20.0, 15.0, 0.5)

    def test_De(self):
        h = 1.0e-32
        eps = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        acted = self.YU.elastic.De
        vectors = []
        for i in range(6):
            add_complex_v = np.zeros(6, dtype=complex)
            add_complex_v[i] += h * 1.0j
            eps_comp = eps + add_complex_v
            sig_comp = self.YU.elastic.De @ eps_comp
            exp_n = np.imag(sig_comp) / h
            vectors.append(exp_n)
        expected = np.vstack(vectors)
        assert np.allclose(acted, expected)

    def test_De_inv(self):
        h = 1.0e-32
        sig = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        acted = self.YU.elastic.De_inv
        vectors = []
        for i in range(6):
            add_complex_v = np.zeros(6, dtype=complex)
            add_complex_v[i] += h * 1.0j
            sig_comp = sig + add_complex_v
            eps_comp = self.YU.elastic.De_inv @ sig_comp
            v_n = np.imag(eps_comp) / h
            vectors.append(v_n)
        expected = np.vstack(vectors)
        assert np.allclose(acted, expected)
    
    def test_norm(self):
        h = 1.0e-32
        sig = np.array([10, 20, 30, 40, 50, 60])
        fro_norm = np.sqrt((np.diag([1, 1, 1, 2, 2, 2]) @ sig) @ sig)
        acted = sig / fro_norm
        expected = np.zeros(6)
        for i in range(6):
            add_complex_v = np.zeros(6, dtype=complex)
            add_complex_v[i] += h * 1.0j
            sig_comp = sig + add_complex_v
            v_comp = np.sqrt((np.diag([1, 1, 1, 2, 2, 2]) @ sig_comp) @ sig_comp)
            expected[i] = v_comp.imag / h
        assert np.allclose(acted, expected)
        

    def test_f_f_dsig(self):
        # arrange
        h = 1.0e-32
        sig = np.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        beta = np.array([0.0, 0.0, 0.0, 20.0, 30.0, 50.0])
        theta = np.array([-5, 10, -5, 10.0, 5.0, 10.0])
        sig_d = Id_s @ sig
        eta = sig_d - beta - theta
        f_f_dsig_expected = np.zeros(6)
        for i in range(6):
            add_complex_v = np.zeros(6, dtype=complex)
            add_complex_v[i] += h * 1.0j
            sig_d_comp = sig_d + add_complex_v * np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
            eta_d_comp = sig_d_comp - beta - theta
            f_f_i = self.YU.calc_f_f(eta_d_comp)
            i_v = f_f_i.imag / h
            f_f_dsig_expected[i] = i_v

        # act
        f_f_dsig_acted = self.YU.calc_f_f_dsig(eta)
        
        # assert
        assert np.allclose(f_f_dsig_expected, f_f_dsig_acted)

    def test_f_f_dbeta(self):
         # arrange
        h = 1.0e-32
        sig = np.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        beta = np.array([0.0, 0.0, 0.0, 20.0, 0.0, 0.0])
        theta = np.array([-5, 10, -5, 0.0, 0.0, 0.0])
        sig_d = Id_s @ sig
        eta = sig_d - beta - theta
        f_f_dbeta_expected = np.zeros(6)
        for i in range(6):
            add_complex_v = np.zeros(6, dtype=complex)
            add_complex_v[i] += h * 1.0j
            beta_comp = beta + add_complex_v * np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])
            eta_d_comp = sig_d - beta_comp - theta
            f_f_i = self.YU.calc_f_f(eta_d_comp)
            i_v = f_f_i.imag / h
            f_f_dbeta_expected[i] = i_v

        # act
        f_f_dbeta_acted = self.YU.calc_f_f_dbeta(eta)
        
        # assert
        assert np.allclose(f_f_dbeta_expected, f_f_dbeta_acted)


    def test_f_f_dtheta(self):
         # arrange
        h = 1.0e-32
        sig = np.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        beta = np.array([0.0, 0.0, 0.0, 20.0, 0.0, 0.0])
        theta = np.array([-5, 10, -5, 0.0, 0.0, 0.0])
        sig_d = Id_s @ sig
        eta = sig_d - beta - theta
        f_f_dbeta_expected = np.zeros(6)
        for i in range(6):
            add_complex_v = np.zeros(6, dtype=complex)
            add_complex_v[i] += h * 1.0j
            theta_comp = theta + add_complex_v * np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])
            eta_d_comp = sig_d - beta - theta_comp
            f_f_i = self.YU.calc_f_f(eta_d_comp)
            i_v = f_f_i.imag / h
            f_f_dbeta_expected[i] = i_v

        # act
        f_f_dbeta_acted = self.YU.calc_f_f_dtheta(eta)
        
        # assert
        assert np.allclose(f_f_dbeta_expected, f_f_dbeta_acted)

    def test_f_ep_dsig(self):
         # arrange
        h = 1.0e-32
        sig = np.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        delta_gam = 1.5e-5
        sig_tri = sig + self.YU.elastic.De @ (np.array([1.0, -0.5, -0.5, 0.0, 0.0, 0.0]) * delta_gam)
        beta = np.array([6.0, 6.0, -12.0, 0.0, 0.0, 0.0])
        theta = np.array([-5, 10, -5, 0.0, 0.0, 0.0])
        sig_d = Id_s @ sig
        sig_d_tri = Id_s @ sig_tri
        eta = sig_d - beta - theta
        g, n_s = self.YU.calc_g(eta)
        vectors = []
        for i in range(6):
            add_complex_v = np.zeros(6, dtype=complex)
            add_complex_v[i] += h * 1.0j
            sig_d_comp = sig_d + add_complex_v * np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])
            eta_comp = sig_d_comp - beta - theta
            g_comp, n_s_comp = self.YU.calc_g(eta_comp)
            f_ep_dsig_comp = np.imag(self.YU.calc_f_ep(sig_d_comp, sig_d_tri, n_s_comp, delta_gam)) / h
            vectors.append(f_ep_dsig_comp)
        f_ep_dsig_expected = np.vstack(vectors)

        # act
        f_ep_dsig_acted = self.YU.calc_f_ep_dsig(g, n_s, delta_gam)

        # assert
        assert np.allclose(f_ep_dsig_acted, f_ep_dsig_expected)

    def test_f_ep_dbeta(self):
         # arrange
        h = 1.0e-32
        sig = np.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        delta_gam = 1.5e-5
        sig_tri = sig + self.YU.elastic.De @ (np.array([1.0, -0.5, -0.5, 0.0, 0.0, 0.0]) * delta_gam)
        beta = np.array([0.0, 0.0, 0.0, 10.0, 0.0, 0.0])
        theta = np.array([-5, 10, -5, 0.0, 0.0, 0.0])
        sig_d = Id_s @ sig
        sig_d_tri = Id_s @ sig_tri
        eta = sig_d - beta - theta
        g, n_s = self.YU.calc_g(eta)
        vectors = []
        for i in range(6):
            add_complex_v = np.zeros(6, dtype=complex)
            add_complex_v[i] += h * 1.0j
            beta_comp = beta + add_complex_v * np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])
            eta_comp = sig_d - beta_comp - theta
            g_comp, n_s_comp = self.YU.calc_g(eta_comp)
            f_ep_dbeta_comp = np.imag(self.YU.calc_f_ep(sig_d, sig_d_tri, n_s_comp, delta_gam)) / h
            vectors.append(f_ep_dbeta_comp)
        f_ep_dbeta_expected = np.vstack(vectors)

        # act
        f_ep_dbeta_acted = self.YU.calc_f_ep_dbeta(g, n_s, delta_gam)

        # assert
        assert np.allclose(f_ep_dbeta_acted, f_ep_dbeta_expected)

    def test_f_ep_dtheta(self):
        # arrange
        h = 1.0e-32
        sig = np.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        delta_gam = 1.0e-5
        sig_tri = sig + self.YU.elastic.De @ (np.array([1.0, -0.5, -0.5, 0.0, 0.0, 0.0]) * delta_gam)
        beta = np.array([0.0, 0.0, 0.0, 30.0, 30.0, 30.0])
        theta = np.array([-5, 10, -5, 0.0, 0.0, 0.0])
        sig_d = Id_s @ sig
        sig_d_tri = Id_s @ sig_tri
        eta = sig_d - beta - theta
        g, n_s = self.YU.calc_g(eta)
        vectors = []
        for i in range(6):
            add_complex_v = np.zeros(6, dtype=complex)
            add_complex_v[i] += h * 1.0j
            theta_comp = theta + add_complex_v * np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
            eta_comp = sig_d - beta - theta_comp
            g_comp, n_s_comp = self.YU.calc_g(eta_comp)
            f_ep_dsig_comp = np.imag(self.YU.calc_f_ep(sig_d, sig_d_tri, n_s_comp, delta_gam)) / h
            vectors.append(f_ep_dsig_comp)
        f_ep_dsig_expected = np.vstack(vectors)

        # act
        f_ep_dsig_acted = self.YU.calc_f_ep_dtheta(g, n_s, delta_gam)

        # assert
        print(f_ep_dsig_acted / f_ep_dsig_expected)

    def test_f_beta_dsig(self):
        # arrange
        h = 1.0e-32
        sig = np.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        delta_gam = 1.5e-5
        sig_tri = sig + self.YU.elastic.De @ (np.array([1.0, -0.5, -0.5, 0.0, 0.0, 0.0]) * delta_gam)
        self.YU.beta = np.array([0.0, 0.0, 0.0, 30.0, 30.0, 30.0])
        delta_beta = np.array([0.0, 0.0, 0.0, 1.0, 0.5, 1.0])
        theta = np.array([-5, 10, -5, 0.0, 0.0, 0.0])
        sig_d = Id_s @ sig
        sig_d_tri = Id_s @ sig_tri
        eta = sig_d - (self.YU.beta + delta_beta) - theta
        g, n_s = self.YU.calc_g(eta)
        vectors = []
        for i in range(6):
            add_complex_v = np.zeros(6, dtype=complex)
            add_complex_v[i] += h * 1.0j
            sig_d_comp = sig_d + add_complex_v * np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
            eta_comp = sig_d_comp - (self.YU.beta + delta_beta) - theta
            g_comp, n_s_comp = self.YU.calc_g(eta_comp)
            v = np.imag(self.YU.calc_f_beta(eta_comp, delta_beta, delta_gam)) / h
            vectors.append(v)
        expected = np.vstack(vectors)

        # act
        acted = self.YU.calc_f_beta_dsig(delta_gam)

        # assert
        print(acted / expected)
        assert np.allclose(acted, expected)

    def test_f_beta_dbeta(self):
        # arrange
        h = 1.0e-32
        sig = np.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        delta_gam = 1.5e-5
        sig_tri = sig + self.YU.elastic.De @ (np.array([1.0, -0.5, -0.5, 0.0, 0.0, 0.0]) * delta_gam)
        self.YU.beta = np.array([0.0, 0.0, 0.0, 30.0, 30.0, 30.0])
        delta_beta = np.array([0.0, 0.0, 0.0, 1.0, 0.5, 1.0])
        theta = np.array([-5, 10, -5, 0.0, 0.0, 0.0])
        sig_d = Id_s @ sig
        sig_d_tri = Id_s @ sig_tri
        eta = sig_d - (self.YU.beta + delta_beta) - theta
        g, n_s = self.YU.calc_g(eta)
        vectors = []
        for i in range(6):
            add_complex_v = np.zeros(6, dtype=complex)
            add_complex_v[i] += h * 1.0j
            delta_beta_comp = delta_beta + add_complex_v * np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
            eta_comp = sig_d - (self.YU.beta + delta_beta_comp) - theta
            v = np.imag(self.YU.calc_f_beta(eta_comp, delta_beta_comp, delta_gam)) / h
            vectors.append(v)
        expected = np.vstack(vectors)

        # act
        acted = self.YU.calc_f_beta_dbeta(delta_gam)

        # assert
        print(acted / expected)
        assert np.allclose(acted, expected)

    def test_f_beta_dtheta(self):
        # arrange
        h = 1.0e-32
        sig = np.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        delta_gam = 1.5e-5
        sig_tri = sig + self.YU.elastic.De @ (np.array([1.0, -0.5, -0.5, 0.0, 0.0, 0.0]) * delta_gam)
        self.YU.beta = np.array([0.0, 0.0, 0.0, 30.0, 30.0, 30.0])
        delta_beta = np.array([0.0, 0.0, 0.0, 1.0, 0.5, 1.0])
        theta = np.array([-5, 10, -5, 0.0, 0.0, 0.0])
        sig_d = Id_s @ sig
        sig_d_tri = Id_s @ sig_tri
        eta = sig_d - (self.YU.beta + delta_beta) - theta
        g, n_s = self.YU.calc_g(eta)
        vectors = []
        for i in range(6):
            add_complex_v = np.zeros(6, dtype=complex)
            add_complex_v[i] += h * 1.0j
            theta_comp = theta + add_complex_v * np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
            eta_comp = sig_d - (self.YU.beta + delta_beta) - theta_comp
            v = np.imag(self.YU.calc_f_beta(eta_comp, delta_beta, delta_gam)) / h
            vectors.append(v)
        expected = np.vstack(vectors)

        # act
        acted = self.YU.calc_f_beta_dtheta(delta_gam)

        # assert
        print(acted / expected)
        assert np.allclose(acted, expected)

    def test_f_beta_dgamma(self):
        # arrange
        h = 1.0e-32
        sig = np.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        delta_gam = 1.5e-5
        sig_tri = sig + self.YU.elastic.De @ (np.array([1.0, -0.5, -0.5, 0.0, 0.0, 0.0]) * delta_gam)
        self.YU.beta = np.array([0.0, 0.0, 0.0, 30.0, 30.0, 30.0])
        delta_beta = np.array([0.0, 0.0, 0.0, 1.0, 0.5, 1.0])
        theta = np.array([-5, 10, -5, 0.0, 0.0, 0.0])
        sig_d = Id_s @ sig
        sig_d_tri = Id_s @ sig_tri
        eta = sig_d - (self.YU.beta + delta_beta) - theta
        g, n_s = self.YU.calc_g(eta)
        eta = sig_d - (self.YU.beta + delta_beta) - theta
        expected = np.imag(self.YU.calc_f_beta(eta, delta_beta, (delta_gam + h * 1.0j))) / h
        
        # act
        acted = self.YU.calc_f_beta_dgamma(eta, delta_beta)

        # assert
        assert np.allclose(acted, expected)
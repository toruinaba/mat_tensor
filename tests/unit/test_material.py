import numpy as np
from src.material import Elastic, Linear_isotropic, AF_kinematic, Chaboche_n
from src.util import Id_s

NTENS = 3
E = 205000.0
NU = 0.3

class Test_elastic:
    def test_De(self):
        # arange
        De_expect = np.zeros([6, 6])
        lame1 = NU * E / (1 + NU) / (1 - 2 * NU)
        lame2 = E / 2 / (1 + NU) # = G
        for i_m in range(NTENS):
            De_expect[i_m, i_m] += 2 * lame2
            for i_n in range(NTENS):
                De_expect[i_m, i_n] += lame1
        for i_m in range(NTENS, 2 * NTENS):
            De_expect[i_m, i_m] += lame2

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
        lame1 = NU * E / (1 + NU) / (1 - 2 * NU)
        lame2 = E / 2 / (1 + NU)
        self.MAT.initialize()

        sig = np.array([300.0, 5.0, 30.0, 10.0, 20.0, 50.0])
        sig_d = Id_s @ sig
        q_tri, n_bar = self.MAT.calc_tri(sig_d)
        del_gam = 1.0e-5
        self.MAT.update_i(del_gam, n_bar)
        self.MAT.update()
        f_ip1_prime = self.MAT.calc_f_ip1_prime(q_tri, del_gam, n_bar)
        eflame1 = 2 * lame2 * lame2 * del_gam / q_tri
        eflame2 = 3 * eflame1
        eflame3 = 0.5 * eflame2
        ghard = 6 * lame2 * lame2 * (del_gam / q_tri - 1 / f_ip1_prime)

        for i_m in range(NTENS):
            Dep_expect[i_m, i_m] += 2 * lame2 - eflame2
            for i_n in range(NTENS):
                Dep_expect[i_m, i_n] += lame1 + eflame1
        for i_m in range(NTENS, 2 * NTENS):
            Dep_expect[i_m, i_m] += lame2 - eflame3
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
        lame1 = NU * E / (1 + NU) / (1 - 2 * NU)
        lame2 = E / 2 / (1 + NU)
        self.MAT.initialize()

        sig = np.array([300.0, 5.0, 30.0, 10.0, 20.0, 50.0])
        sig_d = Id_s @ sig
        q_tri, n_bar = self.MAT.calc_tri(sig_d)
        del_gam = 1.0e-3
        self.MAT.update_i(del_gam, n_bar)
        self.MAT.update()
        theta = 1 / (1 + self.MAT.k * del_gam)
        f_ip1_prime = self.MAT.calc_f_ip1_prime(q_tri, del_gam, n_bar)
        eflame1 = 2 * lame2 * lame2 * del_gam / q_tri
        eflame2 = 3 * eflame1
        eflame3 = 0.5 * eflame2
        ghard1 = 6 * lame2 * lame2 * (del_gam / q_tri - 1 / f_ip1_prime)
        ghard2 = 3 * np.sqrt(6) * lame2 * lame2 * self.MAT.k * theta * theta * del_gam / (q_tri * f_ip1_prime)


        n4d = np.zeros((6, 6))
        for i_m in range(2 * NTENS):
            n4d[i_m, i_m] += 1.0
            for i_n in range(2 * NTENS):
                n4d[i_m, i_n] -= n_bar[i_m] * n_bar[i_n]

        n4d_alpha = np.zeros(6)
        for i_m in range(2 * NTENS):
            for i_n in range(2 * NTENS):
                n4d_alpha[i_m] += n4d[i_m, i_n] * self.MAT.alpha[i_n]

        for i_m in range(NTENS):
            Dep_expect[i_m, i_m] += 2 * lame2 - eflame2
            for i_n in range(NTENS):
                Dep_expect[i_m, i_n] += lame1 + eflame1
        for i_m in range(NTENS, 2 * NTENS):
            Dep_expect[i_m, i_m] += lame2 - eflame3
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
        lame1 = NU * E / (1 + NU) / (1 - 2 * NU)
        lame2 = E / 2 / (1 + NU)
        self.MAT.initialize()

        sig = np.array([300.0, 5.0, 30.0, 10.0, 20.0, 50.0])
        sig_d = Id_s @ sig
        q_tri, n_bar = self.MAT.calc_tri(sig_d)
        del_gam = 1.0e-3
        self.MAT.update_i(del_gam, n_bar)
        self.MAT.update()
        f_ip1_prime = self.MAT.calc_f_ip1_prime(q_tri, del_gam, n_bar)
        eflame1 = 2 * lame2 * lame2 * del_gam / q_tri
        eflame2 = 3 * eflame1
        eflame3 = 0.5 * eflame2
        ghard1 = 6 * lame2 * lame2 * (del_gam / q_tri - 1 / f_ip1_prime)
        ghard2 = 3 * np.sqrt(6) * lame2 * lame2 * del_gam / (q_tri * f_ip1_prime)


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
            Dep_expect[i_m, i_m] += 2 * lame2 - eflame2
            for i_n in range(NTENS):
                Dep_expect[i_m, i_n] += lame1 + eflame1
        for i_m in range(NTENS, 2 * NTENS):
            Dep_expect[i_m, i_m] += lame2 - eflame3
        for i_m in range(2 * NTENS):
            for i_n in range(2 * NTENS):
                Dep_expect[i_m, i_n] += ghard1 * n_bar[i_m] * n_bar[i_n] - ghard2 * n4d_alpha[i_m] * n_bar[i_n]

        # act
        Dep_act = self.MAT.calc_Dep(q_tri, del_gam, n_bar)
        # assert
        assert np.allclose(Dep_act, Dep_expect, atol=1.0e-6)
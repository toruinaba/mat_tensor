import numpy as np
import pytest
from src.shell.utils import I_sh, Is_sh, Id_s_sh, P
from src.shell.materials import Elastic_sh, Yoshida_uemori_sh2


class Test_yoshida_uemori:
    YU = Yoshida_uemori_sh2(
        Elastic_sh(205000.0, 0.3),
        124.0,
        168.0,
        500.0,
        190.0,
        12.0,
        9.0,
        0.5,
        205000.0,
        0.0,
    )

    def test_f_f_dsig(self):
        # arrange
        h = 1.0e-32
        sig = np.array([100.0, 0.0, 0.0])
        beta = np.array([-10.0, 10.0, 0.0])
        theta = np.array([10, -10, 5])
        eta = sig - beta - theta
        f_f_dsig_expected = np.zeros(3)
        for i in range(3):
            add_complex_v = np.zeros(3, dtype=complex)
            add_complex_v[i] += h * 1.0j
            sig_comp = sig + add_complex_v
            eta_comp = sig_comp - beta - theta
            f_f_i = self.YU.calc_f_f(eta_comp)
            i_v = f_f_i.imag / h
            f_f_dsig_expected[i] = i_v

        # act
        f_f_dsig_acted = self.YU.calc_f_f_dsig(eta)

        # assert
        assert np.allclose(f_f_dsig_expected, f_f_dsig_acted)

    def test_f_f_dtheta(self):
        # arrange
        h = 1.0e-32
        sig = np.array([100.0, 0.0, 0.0])
        beta = np.array([-10.0, 10.0, 0.0])
        theta = np.array([10, -10, 5])
        eta = sig - beta - theta
        f_f_dsig_expected = np.zeros(3)
        for i in range(3):
            add_complex_v = np.zeros(3, dtype=complex)
            add_complex_v[i] += h * 1.0j
            theta_comp = theta + add_complex_v
            eta_comp = sig - beta - theta_comp
            f_f_i = self.YU.calc_f_f(eta_comp)
            i_v = f_f_i.imag / h
            f_f_dsig_expected[i] = i_v

        # act
        f_f_dsig_acted = self.YU.calc_f_f_dtheta(eta)

        # assert
        assert np.allclose(f_f_dsig_expected, f_f_dsig_acted)

    def test_f_f_dbeta(self):
        # arrange
        h = 1.0e-32
        sig = np.array([100.0, 0.0, 0.0])
        beta = np.array([-10.0, 10.0, 0.0])
        theta = np.array([10, -10, 5])
        eta = sig - beta - theta
        f_f_dsig_expected = np.zeros(3)
        for i in range(3):
            add_complex_v = np.zeros(3, dtype=complex)
            add_complex_v[i] += h * 1.0j
            beta_comp = beta + add_complex_v
            eta_comp = sig - beta_comp - theta
            f_f_i = self.YU.calc_f_f(eta_comp)
            i_v = f_f_i.imag / h
            f_f_dsig_expected[i] = i_v

        # act
        f_f_dsig_acted = self.YU.calc_f_f_dtheta(eta)

        # assert
        assert np.allclose(f_f_dsig_expected, f_f_dsig_acted)

    def test_f_ep_dsig(self):
        # arrange
        h = 1.0e-32
        sig = np.array([100.0, 0.0, 0.0])
        sig_tri = np.array([105.0, 0.0, 0.0])
        beta = np.array([-10.0, 10.0, 0.0])
        theta = np.array([10, -10, 5])
        delta_gam = 1.0e-4
        eta = sig - beta - theta
        vectors = []
        for i in range(3):
            add_complex_v = np.zeros(3, dtype=complex)
            add_complex_v[i] += h * 1.0j
            sig_comp = sig + add_complex_v
            eta_comp = sig_comp - beta - theta
            f_f_i = self.YU.calc_f_ep(sig_comp, eta_comp, sig_tri, delta_gam)
            i_v = f_f_i.imag / h
            vectors.append(i_v)
        expected = np.vstack(vectors)

        # act
        f_f_dsig_acted = self.YU.calc_f_ep_dsig(eta, delta_gam)

        # assert
        assert np.allclose(expected, f_f_dsig_acted)

    def test_f_ep_dtheta(self):
        # arrange
        h = 1.0e-32
        sig = np.array([100.0, 0.0, 0.0])
        sig_tri = np.array([105.0, 0.0, 0.0])
        beta = np.array([-10.0, 10.0, 0.0])
        theta = np.array([10, -10, 5])
        delta_gam = 1.0e-4
        eta = sig - beta - theta
        vectors = []
        for i in range(3):
            add_complex_v = np.zeros(3, dtype=complex)
            add_complex_v[i] += h * 1.0j
            theta_comp = theta + add_complex_v
            eta_comp = sig - beta - theta_comp
            f_f_i = self.YU.calc_f_ep(sig, eta_comp, sig_tri, delta_gam)
            i_v = f_f_i.imag / h
            vectors.append(i_v)
        expected = np.vstack(vectors)
        # act
        f_f_dsig_acted = self.YU.calc_f_ep_dtheta(eta, delta_gam)
        # assert
        assert np.allclose(expected, f_f_dsig_acted)

    def test_f_ep_dbeta(self):
        # arrange
        h = 1.0e-32
        sig = np.array([100.0, 0.0, 0.0])
        sig_tri = np.array([105.0, 0.0, 0.0])
        beta = np.array([-10.0, 10.0, 0.0])
        theta = np.array([10, -10, 5])
        delta_gam = 1.0e-4
        eta = sig - beta - theta
        vectors = []
        for i in range(3):
            add_complex_v = np.zeros(3, dtype=complex)
            add_complex_v[i] += h * 1.0j
            beta_comp = beta + add_complex_v
            eta_comp = sig - beta_comp - theta
            f_f_i = self.YU.calc_f_ep(sig, eta_comp, sig_tri, delta_gam)
            i_v = f_f_i.imag / h
            vectors.append(i_v)
        expected = np.vstack(vectors)
        # act
        f_f_dsig_acted = self.YU.calc_f_ep_dbeta(eta, delta_gam)
        # assert
        assert np.allclose(expected, f_f_dsig_acted)

    def test_f_ep_dgamma(self):
        # arrange
        h = 1.0e-32
        sig = np.array([100.0, 0.0, 0.0])
        sig_tri = np.array([105.0, 0.0, 0.0])
        beta = np.array([-0.0, 10.0, 0.0])
        theta = np.array([10, -10, 5])
        delta_gam = 1.0e-4
        eta = sig - beta - theta
        delta_gam_comp = delta_gam + h * 1.0j
        f_ep_i = self.YU.calc_f_ep(sig, eta, sig_tri, delta_gam_comp)
        f_f_dsig_expected = np.imag(f_ep_i) / h

        # act
        f_ep_dgamma_acted = self.YU.calc_f_ep_dgamma(eta, delta_gam)

        # assert
        assert np.allclose(f_f_dsig_expected, f_ep_dgamma_acted)

    def test_f_theta_dsig(self):
        # arrange
        h = 1.0e-32
        sig = np.array([100.0, 0.0, 0.0])
        sig_tri = np.array([105.0, 0.0, 0.0])
        beta = np.array([-10.0, 10.0, 0.0])
        theta = np.array([10, -10, 5])
        delta_gam = 1.0e-4
        self.YU.R = 10.0
        s = 1 / (1 + 2 / 3 * self.YU.k * self.YU.sig_y * delta_gam)
        R = s * (
            self.YU.R + 2 / 3 * self.YU.k * self.YU.sig_y * self.YU.Rsat * delta_gam
        )
        a = self.YU.B + R - self.YU.sig_y
        vectors = []
        for i in range(3):
            add_complex_v = np.zeros(3, dtype=complex)
            add_complex_v[i] += h * 1.0j
            sig_comp = sig + add_complex_v
            eta_comp = sig_comp - beta - theta
            f_f_i = self.YU.calc_f_theta(eta_comp, theta, a, delta_gam)
            i_v = f_f_i.imag / h
            vectors.append(i_v)
        expected = np.vstack(vectors)

        # act
        f_theta_dsig_acted = self.YU.calc_f_theta_dsig(a, delta_gam)

        # assert
        assert np.allclose(expected, f_theta_dsig_acted)

    # @pytest.mark.skip(reason="調査中")
    def test_f_theta_dtheta(self):
        # arrange
        h = 1.0e-32
        sig = np.array([100.0, 0.0, 0.0])
        beta = np.array([-10.0, 10.0, 0.0])
        theta = np.array([10, -10, 5])
        delta_gam = 1.0e-4
        eta = sig - beta - theta
        self.YU.R = 10.0
        s = 1 / (1 + 2 / 3 * self.YU.k * self.YU.sig_y * delta_gam)
        R = s * (
            self.YU.R + 2 / 3 * self.YU.k * self.YU.sig_y * self.YU.Rsat * delta_gam
        )
        a = self.YU.B + R - self.YU.sig_y
        vectors = []
        for i in range(3):
            add_complex_v = np.zeros(3, dtype=complex)
            add_complex_v[i] += h * 1.0j
            theta_comp = theta + add_complex_v
            eta_comp = sig - beta - theta_comp
            f_f_i = self.YU.calc_f_theta(eta_comp, theta_comp, a, delta_gam)
            i_v = f_f_i.imag / h
            vectors.append(i_v)
        expected = np.vstack(vectors)
        # act
        f_theta_dsig_acted = self.YU.calc_f_theta_dtheta(eta, theta, a, delta_gam)
        # assert
        assert np.allclose(expected, f_theta_dsig_acted)

    def test_f_theta_dbeta(self):
        # arrange
        h = 1.0e-32
        sig = np.array([100.0, 0.0, 0.0])
        sig_tri = np.array([105.0, 0.0, 0.0])
        beta = np.array([-10.0, 10.0, 0.0])
        theta = np.array([10, -10, 5])
        delta_gam = 1.0e-4
        eta = sig - beta - theta
        self.YU.R = 10.0
        s = 1 / (1 + 2 / 3 * self.YU.k * self.YU.sig_y * delta_gam)
        R = s * (
            self.YU.R + 2 / 3 * self.YU.k * self.YU.sig_y * self.YU.Rsat * delta_gam
        )
        a = self.YU.B + R - self.YU.sig_y
        vectors = []
        for i in range(3):
            add_complex_v = np.zeros(3, dtype=complex)
            add_complex_v[i] += h * 1.0j
            beta_comp = beta + add_complex_v
            eta_comp = sig - beta_comp - theta
            f_f_i = self.YU.calc_f_theta(eta_comp, theta, a, delta_gam)
            i_v = f_f_i.imag / h
            vectors.append(i_v)
        expected = np.vstack(vectors)

        # act
        f_theta_dbeta_acted = self.YU.calc_f_theta_dbeta(a, delta_gam)

        # assert
        assert np.allclose(expected, f_theta_dbeta_acted)

    @pytest.mark.skip(reason="調査中")
    def test_f_theta_dgamma(self):
        # arrange
        h = 1.0e-32
        sig = np.array([100.0, 10.0, 0.0])
        beta = np.array([-10, 10.0, 5.0])
        delta_gam = 1.5e-4
        self.YU.R = 10.0
        self.YU.theta = np.array([10, -10, 5.0])
        theta_i = np.array([15, -15, 6.0])
        eta = sig - beta - theta_i
        s = 1 / (1 + 2 / 3 * self.YU.k * self.YU.sig_y * delta_gam)
        R = s * (
            self.YU.R + 2 / 3 * self.YU.k * self.YU.sig_y * self.YU.Rsat * delta_gam
        )
        a = self.YU.B + R - self.YU.sig_y

        delta_gam_comp = delta_gam + h * 1.0j
        s_comp = 1 / (1 + 2 / 3 * self.YU.k * self.YU.sig_y * delta_gam_comp)
        R_comp = s_comp * (
            self.YU.R
            + 2 / 3 * self.YU.k * self.YU.sig_y * self.YU.Rsat * delta_gam_comp
        )
        a_comp = self.YU.B + R_comp - self.YU.sig_y
        f_theta_i = self.YU.calc_f_theta(eta, theta_i, a_comp, delta_gam_comp)
        f_theta_dgamma_expected = np.imag(f_theta_i) / h
        # act
        f_theta_dgamma_acted = self.YU.calc_f_theta_dgamma(eta, theta_i, a, delta_gam)
        # assert
        assert np.allclose(f_theta_dgamma_expected, f_theta_dgamma_acted)

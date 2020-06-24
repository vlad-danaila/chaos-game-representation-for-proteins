import unittest
import portion as p
from tobit.tobit_optimization import tobit_mean_and_variance_reparametrization

DISABLE_LONG_RUNNING_TESTS = True

class TobitOptimizationTest(unittest.TestCase):

    def check_mean_std(self, ic50, expected_mean, expected_std, delta_real = .4, delta_aprox = .5):
        mean, std = tobit_mean_and_variance_reparametrization(ic50, aproximation = False)
        self.assertAlmostEqual(mean.item(), expected_mean, delta = delta_real)
        self.assertAlmostEqual(std.item(), expected_std, delta = delta_real)

        mean, std = tobit_mean_and_variance_reparametrization(ic50, aproximation = True)
        self.assertAlmostEqual(mean.item(), expected_mean, delta = delta_aprox)
        self.assertAlmostEqual(std.item(), expected_std, delta = delta_aprox)

    # 1) 20 30 40
    def test_single_valued_only(self):
        ic50 = [p.singleton(20), p.singleton(30), p.singleton(40)]
        self.check_mean_std(ic50, 30, 8.1650)

    # 2) 30 >50 >50
    def test_right_censored_30_50_50(self):
        ic50 = [p.singleton(30), p.closed(50, p.inf), p.closed(50, p.inf)]
        self.check_mean_std(ic50, 58.0191, 23.6987)

    # 3) 30 >50 >50 >50 >50
    def test_right_censored_30_50_50_50_50(self):
        ic50 = [p.singleton(30), p.closed(50, p.inf), p.closed(50, p.inf), p.closed(50, p.inf), p.closed(50, p.inf)]
        self.check_mean_std(ic50, 73.3895, 29.4739)

    # 4) 30 30 >50
    def test_right_censored_30_30_50(self):
        ic50 = [p.singleton(30), p.singleton(30), p.closed(50, p.inf)]
        self.check_mean_std(ic50, 39.2030, 13.5950)

    # 5) 30 30 30 30 >50
    def test_right_censored_30_30_30_30_50(self):
        ic50 = [p.singleton(30), p.singleton(30), p.singleton(30), p.singleton(30), p.closed(50, p.inf)]
        self.check_mean_std(ic50, 34.8353, 9.8508)

    # 6) 50 >30 >30
    def test_right_censored_50_30_30(self):
        if not DISABLE_LONG_RUNNING_TESTS:
            ic50 = [p.singleton(50), p.closed(30, p.inf), p.closed(30, p.inf)]
            self.check_mean_std(ic50, 49.9330, 0.3652)

    # 7) 40 50 >30 >30
    def test_right_censored_40_50_30_30(self):
        ic50 = [p.singleton(50), p.singleton(40), p.closed(30, p.inf), p.closed(30, p.inf)]
        self.check_mean_std(ic50, 45.0124, 4.9814)

    # 8) >30 >30 >30
    def test_all_right_censored(self):
        if not DISABLE_LONG_RUNNING_TESTS:
            ic50 = [p.closed(30, p.inf), p.closed(30, p.inf), p.closed(30, p.inf)]
            self.check_mean_std(ic50, 30, 1e-10)

    # 9) <10 <10 30
    def test_left_censored_10_10_30(self):
        ic50 = [p.closed(-p.inf, 10), p.closed(-p.inf, 10), p.singleton(30)]
        self.check_mean_std(ic50, 1.8617, 23.7228)

    # 10) <10 <10 <10 <10 30
    def test_left_censored_10_10_10_10_30(self):
        ic50 = [ p.closed(-p.inf, 10), p.closed(-p.inf, 10), p.closed(-p.inf, 10), p.closed(-p.inf, 10), p.singleton(30)]
        self.check_mean_std(ic50, -13.5183, 29.5021)

    # 11) <10 30 30
    def test_left_censored_10_30_30(self):
        ic50 = [p.closed(-p.inf, 10), p.singleton(30), p.singleton(30)]
        self.check_mean_std(ic50, 20.7514, 13.6005)

    # 12) <10 30 30 30 30
    def test_left_censored_10_30_30_30_30(self):
        ic50 = [p.closed(-p.inf, 10), p.singleton(30), p.singleton(30), p.singleton(30), p.singleton(30)]
        self.check_mean_std(ic50, 25.1465, 9.8524)

    # 13) 10 <30 <30
    def test_left_censored_30_30_10(self):
        if not DISABLE_LONG_RUNNING_TESTS:
            ic50 = [p.closed(-p.inf, 30), p.closed(-p.inf, 30), p.singleton(10)]
            self.check_mean_std(ic50, 10.0067, 0.3653)

    # 14) 10 <30 <30 <30 <30
    def test_left_censored_30_30_30_30_10(self):
        if not DISABLE_LONG_RUNNING_TESTS:
            ic50 = [p.closed(-p.inf, 30), p.closed(-p.inf, 30), p.closed(-p.inf, 30), p.closed(-p.inf, 30), p.singleton(10)]
            self.check_mean_std(ic50, 10.0055, 0.3325)

    # 15) 10 20 <30
    def test_left_censored_30_20_10(self):
        ic50 = [p.closed(-p.inf, 30), p.singleton(10), p.singleton(20)]
        self.check_mean_std(ic50, 14.9935, 4.9903)

    # 16) 10 20 <30 <30 <30 <30
    def test_left_censored_30_30_30_30_10_20(self):
        ic50 = [p.closed(-p.inf, 30), p.closed(-p.inf, 30), p.closed(-p.inf, 30), p.closed(-p.inf, 30), p.singleton(10), p.singleton(20)]
        self.check_mean_std(ic50, 14.9774, 4.9659)

    # 17) <30 <30 <30
    def test_left_censored_30_30_30(self):
        ic50 = [p.closed(-p.inf, 30), p.closed(-p.inf, 30), p.closed(-p.inf, 30)]
        self.check_mean_std(ic50, 30.0000, 1.0000e-10)

    # 18) <10 <20 30
    def test_left_censored_10_20_30(self):
        ic50 = [p.closed(-p.inf, 10), p.closed(-p.inf, 20), p.singleton(30)]
        self.check_mean_std(ic50, 7.9252, 18.9532)

    # 19) 30 >40 >50
    def test_right_censored_30_40_50(self):
        ic50 = [p.singleton(30), p.closed(40, p.inf), p.closed(50, p.inf)]
        self.check_mean_std(ic50, 52.0748, 18.9532)

    # 20) >30 40 >50
    def test_right_censored_40_30_50(self):
        ic50 = [p.closed(30, p.inf), p.singleton(40), p.closed(50, p.inf)]
        self.check_mean_std(ic50, 48.4679, 8.7341)

    # 21) >30 40 40 40 >50
    def test_right_censored_30_40_40_40_50(self):
        ic50 = [p.closed(30, p.inf), p.singleton(40), p.singleton(40), p.singleton(40), p.closed(50, p.inf)]
        self.check_mean_std(ic50, 43.1742, 5.5445)

    # 22) >30 40 <50
    def test_left_right_censored_30_40_50(self):
        if DISABLE_LONG_RUNNING_TESTS:
            ic50 = [p.closed(30, p.inf), p.singleton(40), p.closed(-p.inf, 50)]
            self.check_mean_std(ic50, 40, 0.1714)

    # 23) >20 >30 40 <50
    def test_left_right_censored_20_30_40_50(self):
        if DISABLE_LONG_RUNNING_TESTS:
            ic50 = [p.closed(20, p.inf), p.closed(30, p.inf), p.singleton(40), p.closed(-p.inf, 50)]
            self.check_mean_std(ic50, 39.9970, 0.2393)

    # 24) >20 >30 <50
    def test_left_right_censored_20_30_50(self):
        if DISABLE_LONG_RUNNING_TESTS:
            ic50 = [p.closed(20, p.inf), p.closed(30, p.inf), p.closed(-p.inf, 50)]
            self.check_mean_std(ic50, 40.1529, 2.6186)

    # 25) <20 >30
    def test_diverge_20_30(self):
        if DISABLE_LONG_RUNNING_TESTS:
            ic50 = [p.closed(-p.inf, 20), p.closed(30, p.inf)]
            # is the negative std correct ?
            self.check_mean_std(ic50, 25, -1.2508)

    # 26) <20 >25 >50 >50 >50
    def test_diverge_20_25_50_50_50(self):
        if DISABLE_LONG_RUNNING_TESTS:
            ic50 = [p.closed(-p.inf, 20), p.closed(25, p.inf), p.closed(50, p.inf), p.closed(50, p.inf), p.closed(50, p.inf)]
            self.check_mean_std(ic50, 22.5357, -0.9952)

    # 27) >30 35 37 <50
    def test_left_right_censoring_30_35_37_50(self):
        if DISABLE_LONG_RUNNING_TESTS:
            ic50 = [p.closed(30, p.inf), p.singleton(35), p.singleton(37), p.closed(-p.inf, 50)]
            self.check_mean_std(ic50, 36.0000, 1.0001)

    # 28) <30 32 34 >50
    def test_divergent_30_32_34_50(self):
        ic50 = [p.closed(-p.inf, 30), p.singleton(32), p.singleton(34), p.closed(50, p.inf)]
        self.check_mean_std(ic50, 36.0223, 14.4379)
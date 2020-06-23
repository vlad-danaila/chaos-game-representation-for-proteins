import unittest
import portion as p
from tobit.tobit_optimization import tobit_mean_and_variance_reparametrization

DISABLE_LONG_RUNNING_TESTS = False

class TobitOptimizationTest(unittest.TestCase):

    def check_mean_std(self, ic50, expected_mean, expected_std, delta_real = .4, delta_aprox = .5):
        mean, std = tobit_mean_and_variance_reparametrization(ic50, aproximation = False)
        self.assertAlmostEqual(mean.item(), expected_mean, delta = delta_real)
        self.assertAlmostEqual(std.item(), expected_std, delta = delta_real)

        mean, std = tobit_mean_and_variance_reparametrization(ic50, aproximation = True)
        self.assertAlmostEqual(mean.item(), expected_mean, delta = delta_aprox)
        self.assertAlmostEqual(std.item(), expected_std, delta = delta_aprox)

    # 20 30 40
    def test_single_valued_only(self):
        ic50 = [p.singleton(20), p.singleton(30), p.singleton(40)]
        self.check_mean_std(ic50, 30, 8.1650)

    # 30 >50 >50
    def test_right_censored_30_50_50(self):
        ic50 = [p.singleton(30), p.closed(50, p.inf), p.closed(50, p.inf)]
        self.check_mean_std(ic50, 58.0191, 23.6987)

    # 30 >50 >50 >50 >50
    def test_right_censored_30_50_50_50_50(self):
        ic50 = [p.singleton(30), p.closed(50, p.inf), p.closed(50, p.inf), p.closed(50, p.inf), p.closed(50, p.inf)]
        self.check_mean_std(ic50, 73.3895, 29.4739)

    # 30 30 >50
    def test_right_censored_30_30_50(self):
        ic50 = [p.singleton(30), p.singleton(30), p.closed(50, p.inf)]
        self.check_mean_std(ic50, 39.2030, 13.5950)

    # 30 30 30 30 >50
    def test_right_censored_30_30_30_30_50(self):
        ic50 = [p.singleton(30), p.singleton(30), p.singleton(30), p.singleton(30), p.closed(50, p.inf)]
        self.check_mean_std(ic50, 34.8353, 9.8508)

    # 50 >30 >30
    def test_right_censored_50_30_30(self):
        if not DISABLE_LONG_RUNNING_TESTS:
            ic50 = [p.singleton(50), p.closed(30, p.inf), p.closed(30, p.inf)]
            self.check_mean_std(ic50, 49.9330, 1.1563)

    # >30 >30 >30
    def test_all_right_censored(self):
        if not DISABLE_LONG_RUNNING_TESTS:
            ic50 = [p.closed(30, p.inf), p.closed(30, p.inf), p.closed(30, p.inf)]
            self.check_mean_std(ic50, 30, 1e-10)

    # <10 <10 30
    def test_left_censored_10_10_30(self):
        ic50 = [p.closed(-p.inf, 10), p.closed(-p.inf, 10), p.singleton(30)]
        self.check_mean_std(ic50, 1.8617, 23.7228)

    # <10 <10 <10 <10 30
    def test_left_censored_10_10_10_10_30(self):
        ic50 = [ p.closed(-p.inf, 10), p.closed(-p.inf, 10), p.closed(-p.inf, 10), p.closed(-p.inf, 10), p.singleton(30)]
        self.check_mean_std(ic50, -13.5183, 29.5021)
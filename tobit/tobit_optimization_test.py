import unittest
import portion as p
from tobit.tobit_optimization import tobit_mean_and_variance_reparametrization

DISABLE_LONG_RUNNING_TESTS = True

class TobitOptimizationTest(unittest.TestCase):

    def check_mean_std(self, ic50, mean, std, delta_real = 1e-5, delta_aprox = 0.2):
        mean, std = tobit_mean_and_variance_reparametrization(ic50, aproximation=False)
        self.assertAlmostEqual(mean.item(), mean, delta = delta_real)
        self.assertAlmostEqual(std.item(), std, delta = delta_real)

        mean, std = tobit_mean_and_variance_reparametrization(ic50, aproximation=True)
        self.assertAlmostEqual(mean.item(), mean, delta = delta_aprox)
        self.assertAlmostEqual(std.item(), std, delta = delta_aprox)

    # 30 >50 >50
    def test_right_censored_30_50_50(self):
        ic50 = [p.singleton(30), p.closed(50, p.inf), p.closed(50, p.inf)]
        self.check_mean_std(ic50, 58.0191, 23.6987)

    # 30 >50 >50 >50 >50
    def test_right_censored_30_50_50_50_50(self):
        ic50 = [p.singleton(30), p.closed(50, p.inf), p.closed(50, p.inf), p.closed(50, p.inf), p.closed(50, p.inf)]
        self.check_mean_std(ic50, 73.3895, 29.4739)

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
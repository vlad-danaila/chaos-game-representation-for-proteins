from util.data import normalize, unnormalize, to_tensor, to_numpy
import portion as p
import torch as t
from typing import List
import math
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
from tobit.tobit_optimization import read_normalized_tensors_from_assay_intervals, plot_gausian

class LogCDF(t.autograd.Function):

    @staticmethod
    def forward(ctx, x: t.Tensor):
        _x = to_numpy(x)
        pdf = to_tensor(norm.pdf(_x), grad = False)
        cdf = to_tensor(norm.cdf(_x), grad = False)
        ctx.save_for_backward(pdf, cdf)
        return t.log(cdf)

    @staticmethod
    def backward(ctx, grad_output):
        pdf, cdf = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output * pdf / cdf
        return grad_input

log_cdf = LogCDF.apply

def tobit_mean_and_variance_reparametrization(intervals: List[p.interval.Interval]):
    single_val, right_censored, left_censored, data_mean, data_std, N = read_normalized_tensors_from_assay_intervals(intervals)
    delta, gamma = to_tensor(0, grad = True), to_tensor(1, grad = True)
    optimizer = t.optim.SGD([delta, gamma], lr=1e-1)
    patience = 5
    for i in range(10_000):
        prev_delta, prev_gamma = delta.clone(), gamma.clone()
        optimizer.zero_grad()

        # step 1 update based on pdf gradient (for uncensored data)
        # this is the same as -sum(ln(gamma) + ln(pdf(gamma * y - delta)))
        log_likelihood_pdf = -t.sum(t.log(gamma) - ((gamma * single_val - delta) ** 2)/2)
        log_likelihood_pdf.backward()

        # step 2 compute the log(1 - cdf(x)) = log(cdf(-x)) gradient (for right censored data)
        if len(right_censored) > 0:
            log_likelihood_1_minus_cdf = -t.sum(log_cdf(-gamma * right_censored + delta))
            log_likelihood_1_minus_cdf.backward()

        # step 3 compute the log(cdf(x)) gradient (for left censored data)
        if len(left_censored) > 0:
            log_likelihood_cdf = -t.sum(log_cdf(gamma * left_censored - delta))
            log_likelihood_cdf.backward()

        optimizer.step()
        early_stop = math.fabs(delta - prev_delta) + math.fabs(gamma - prev_gamma) < 1e-5
        if early_stop:
            patience -= 1
            if patience == 0:
                break
        else:
            patience = 5

        if i % 100 == 0:
            print(i, delta, gamma)
    print(i, delta, gamma)
    mean, std = delta / gamma, 1 / gamma
    return unnormalize(mean, data_mean, data_std), std * data_std

if __name__ == '__main__':
    no_tobit = np.array([10, 30, 40, 42, 44, 60, 70])
    no_tobit_mean, no_tobit_std = norm.fit(no_tobit)

    ic50 = [p.closed(-p.inf, 10), p.closed(-p.inf, 30), p.singleton(40), p.singleton(42), p.singleton(44), p.closed(60, p.inf), p.closed(70, p.inf)]
    mean, std = tobit_mean_and_variance_reparametrization(ic50, aproximation = False)

    print('No tobit mean', no_tobit_mean, 'std', no_tobit_std)
    plot_gausian(no_tobit_mean, no_tobit_std)
    print('Mean', mean, 'std', std)
    plot_gausian(to_numpy(mean), to_numpy(std))

    plt.show()
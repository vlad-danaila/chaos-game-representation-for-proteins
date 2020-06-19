from data.assay_reader import Assay
import portion as p
import torch as t
from typing import List
import constants
import math
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, pi

# hand crafted
def initial_softplus_aproximation_of_log_1_minus_cdf(x: t.Tensor) -> t.Tensor:
    return -10 * t.log(1 + t.exp(0.7 * x - 2))

# optimized with gradient descent 1
def softplus_aproximation_of_log_1_minus_cdf_v1(x: t.Tensor) -> t.Tensor:
    return -9.4785 * t.log(0.9643 + t.exp(0.7526 * x - 2.1559))

# optimized with gradient descent 2
def softplus_aproximation_of_log_1_minus_cdf_v2(x: t.Tensor) -> t.Tensor:
    return -9.2969 * t.log(0.9753 + t.exp(0.6993 * x - 2.0691))

def fit_softplus_to_log_1_minus_cdf():
    m = t.tensor(-9.2969, dtype=t.float64, requires_grad=True)
    n = t.tensor(0.9753, dtype=t.float64, requires_grad=True)
    p = t.tensor(0.6993, dtype=t.float64, requires_grad=True)
    q = t.tensor(2.0691, dtype=t.float64, requires_grad=True)

    optimizer = t.optim.SGD([m, n, p, q], lr=1e-7)

    for i in range(40_000):
        optimizer.zero_grad()
        x = t.tensor(np.random.uniform(-8, 8, 100), dtype=t.float64, requires_grad=False)
        aprox = m * t.log(n + t.exp(p * x - q))
        expected = np.log(1 - norm.cdf(x))
        expected = t.tensor(expected, dtype=t.float64)
        loss = t.sum((expected - aprox) ** 2)
        loss.backward()
        optimizer.step()
        print(i, loss)
    print(m, n, p, q)
    return lambda x: m * t.log(n + t.exp(p * x - q))

def log_1_minus_cdf_plot(aporx_function):
    x = t.tensor(np.linspace(-6, 6, 1000), dtype=t.float64, requires_grad=False)
    plt.plot(x.clone().detach().numpy(), aporx_function(x).clone().detach().numpy())

if __name__ == '__main__':
    f = fit_softplus_to_log_1_minus_cdf()
    log_1_minus_cdf_plot(lambda x: t.log(1 - t.tensor(norm.cdf(x), dtype=t.float64)))
    log_1_minus_cdf_plot(f)

    plt.show()
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
    m_1 = t.tensor(-9.2969, dtype=t.float64, requires_grad=True)
    n_1 = t.tensor(0.9753, dtype=t.float64, requires_grad=True)
    p_1 = t.tensor(0.6993, dtype=t.float64, requires_grad=True)
    q_1 = t.tensor(2.0691, dtype=t.float64, requires_grad=True)

    m_2 = t.tensor(-9.4785, dtype=t.float64, requires_grad=True)
    n_2 = t.tensor(0.9643, dtype=t.float64, requires_grad=True)
    p_2 = t.tensor(0.7526, dtype=t.float64, requires_grad=True)
    q_2 = t.tensor(2.1559, dtype=t.float64, requires_grad=True)

    r = t.tensor(.5, dtype=t.float64, requires_grad=True)

    optimizer = t.optim.SGD([m_1, n_1, p_1, q_1, m_2, n_2, p_2, q_2, r], lr=1e-5)

    for i in range(40_000):
        optimizer.zero_grad()
        x = t.tensor(np.random.uniform(-5, 5, 1000), dtype=t.float64, requires_grad=False)
        aprox_1 = m_1 * t.log(n_1 + t.exp(p_1 * x - q_1))
        aprox_2 = m_2 * t.log(n_2 + t.exp(p_2 * x - q_2))
        aprox = r * aprox_1 + (1 - r) * aprox_2
        expected = np.log(1 - norm.cdf(x))
        expected = t.tensor(expected, dtype=t.float64)
        loss = t.sum((expected - aprox_1) ** 2)
        loss.backward()
        optimizer.step()
        print(i, loss)
    print(m_1, n_1, p_1, q_1, m_2, n_2, p_2, q_2)
    return lambda x: r * (m_1 * t.log(n_1 + t.exp(p_1 * x - q_1))) + (1 - r) * (m_2 * t.log(n_2 + t.exp(p_2 * x - q_2)))

def log_1_minus_cdf_plot(aporx_function):
    x = t.tensor(np.linspace(-6, 6, 1000), dtype=t.float64, requires_grad=False)
    plt.plot(x.clone().detach().numpy(), aporx_function(x).clone().detach().numpy())

if __name__ == '__main__':
    f = fit_softplus_to_log_1_minus_cdf()
    log_1_minus_cdf_plot(lambda x: t.log(1 - t.tensor(norm.cdf(x), dtype=t.float64)))
    log_1_minus_cdf_plot(f)

    # log_1_minus_cdf_plot(lambda x: t.log(1 - t.tensor(norm.cdf(x), dtype=t.float64)))
    # log_1_minus_cdf_plot(initial_softplus_aproximation_of_log_1_minus_cdf)
    # log_1_minus_cdf_plot(softplus_aproximation_of_log_1_minus_cdf_v2)

    plt.show()
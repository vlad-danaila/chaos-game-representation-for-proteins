import torch as t
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import Module, Parameter
import constants
from util.data import to_tensor
from random import random
from util.timer import timer_start, timer_end

class LogCdfSoftplusAproximation(Module):

    def __init__(self):
        super(LogCdfSoftplusAproximation, self).__init__()
        self.a = Parameter(to_tensor(random()))
        self.b = Parameter(to_tensor(random()))
        self.m = Parameter(to_tensor(-9.2969 + random()))
        self.n = Parameter(to_tensor(0.9753 + random()))
        self.p = Parameter(to_tensor(-0.6993 + random()))
        self.q = Parameter(to_tensor(-2.0691 + random()))

    def forward(self, x):
        return t.sigmoid(self.a * x + self.b) * self.m * t.log(self.n + t.exp(self.p * x + self.q))

class LogCdfEnsembleAproximation(Module):

    def __init__(self, nb_estimators = 10):
        super(LogCdfEnsembleAproximation, self).__init__()
        self.nb_estimators = nb_estimators
        for i in range(nb_estimators):
            estimator_name = 'estimator' + str(i)
            self.add_module(estimator_name, LogCdfSoftplusAproximation())

    def get_estimator(self, index):
        return getattr(self, 'estimator' + str(index))

    def forward(self, x):
        result = self.estimator0.forward(x)
        for i in range(1, self.nb_estimators):
            result = result + self.get_estimator(i).forward(x)
        return result

def fit_softplus_to_log_1_minus_cdf(load_from_chekpoint = False):
    model: t.nn.Module = LogCdfEnsembleAproximation()
    if load_from_chekpoint:
        model.load_state_dict(t.load(constants.LOG_CDF_APROXIMATION_CHECKPOINT))
    optimizer = t.optim.Adam(model.parameters(), lr=1e-5)
    for i in range(50_000):
        optimizer.zero_grad()
        x = t.tensor(np.random.uniform(-30, 30, 10_000), dtype=t.float64, requires_grad=False)
        aprox = model.forward(x)
        expected = np.log(norm.cdf(x))
        expected = t.tensor(expected, dtype=t.float64)
        loss = t.sum(t.abs(expected - aprox))
        loss.backward()
        optimizer.step()
        if i % 1000 == 0:
            print(i, loss)
    print(i, loss)
    t.save(model.state_dict(), constants.LOG_CDF_APROXIMATION_CHECKPOINT)

def log_cdf_plot(aporx_function):
    x = t.tensor(np.linspace(-50, 50, 1000), dtype=t.float64, requires_grad=False)
    plt.plot(x.clone().detach().numpy(), aporx_function(x).clone().detach().numpy())

if __name__ == '__main__':
    fit_softplus_to_log_1_minus_cdf(load_from_chekpoint=True)

    model_log_1_minus_cdf: Module = LogCdfEnsembleAproximation()
    model_log_1_minus_cdf.load_state_dict(t.load(constants.LOG_CDF_APROXIMATION_CHECKPOINT))

    log_cdf_plot(lambda x: t.log(to_tensor(norm.cdf(x))))
    log_cdf_plot(lambda x: model_log_1_minus_cdf.forward(x))

    plt.show()
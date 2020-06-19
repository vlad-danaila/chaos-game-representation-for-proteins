import torch as t
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import Module, Parameter
import constants

class Log1MinusCdfAproximation(Module):

    def __init__(self):
        super(Log1MinusCdfAproximation, self).__init__()
        self.m = Parameter(t.tensor(-9.2969, dtype=t.float64))
        self.n = Parameter(t.tensor(0.9753, dtype=t.float64))
        self.p = Parameter(t.tensor(0.6993, dtype=t.float64))
        self.q = Parameter(t.tensor(2.0691, dtype=t.float64))

    def forward(self, x):
        return self.m * t.log(self.n + t.exp(self.p * x - self.q))

def fit_softplus_to_log_1_minus_cdf():
    model: t.nn.Module = Log1MinusCdfAproximation()
    optimizer = t.optim.SGD(model.parameters(), lr=1e-7)
    for i in range(50_000):
        optimizer.zero_grad()
        x = t.tensor(np.random.uniform(-8, 8, 200), dtype=t.float64, requires_grad=False)
        aprox = model.forward(x)
        expected = np.log(1 - norm.cdf(x))
        expected = t.tensor(expected, dtype=t.float64)
        loss = t.sum(t.abs(expected - aprox))
        loss.backward()
        optimizer.step()
        print(i, loss)
    t.save(model.state_dict(), constants.LOG_1_MINUS_CDF_APROXIMATION_CHECKPOINT)

def log_1_minus_cdf_plot(aporx_function):
    x = t.tensor(np.linspace(-7, 7, 1000), dtype=t.float64, requires_grad=False)
    plt.plot(x.clone().detach().numpy(), aporx_function(x).clone().detach().numpy())

if __name__ == '__main__':
    fit_softplus_to_log_1_minus_cdf()

    model_log_1_minus_cdf: Module = Log1MinusCdfAproximation()
    model_log_1_minus_cdf.load_state_dict(t.load(constants.LOG_1_MINUS_CDF_APROXIMATION_CHECKPOINT))

    log_1_minus_cdf_plot(lambda x: t.log(1 - t.tensor(norm.cdf(x), dtype=t.float64)))
    log_1_minus_cdf_plot(lambda x: model_log_1_minus_cdf.forward(x))

    plt.show()
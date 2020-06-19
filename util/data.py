import torch as t

def normalize(x, mean, std, epsilon = 0):
    return (x - mean) / (std + epsilon)

def to_numpy(tensor: t.Tensor):
    return tensor.clone().detach().numpy()

def to_tensor(x, grad = False):
    return t.tensor(x, dtype=t.float64, requires_grad = grad)
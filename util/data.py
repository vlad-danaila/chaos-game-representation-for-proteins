import torch as t

def normalize(x, mean, std, epsilon = 0):
    if x is None:
        return None
    return (x - mean) / (std + epsilon)

def unnormalize(x, mean, std):
    if x is None:
        return None
    return (x * std) + mean

def to_numpy(tensor: t.Tensor):
    return tensor.clone().detach().numpy()

def to_tensor(x, grad = False):
    return t.tensor(x, dtype=t.float64, requires_grad = grad)
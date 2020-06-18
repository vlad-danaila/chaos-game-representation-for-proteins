def normalize(x, mean, std, epsilon = 0):
    return (x - mean) / (std + epsilon)
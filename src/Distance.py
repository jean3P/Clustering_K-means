import numpy as np



def euclidean(sample, point):
    return np.sqrt(np.sum((sample - point) ** 2))




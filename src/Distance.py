import numpy as np



def euclidean(sample, point):
    assert len(sample) == len(point), "Vectors should have same length"
    return np.sqrt(np.sum((sample - point) ** 2))




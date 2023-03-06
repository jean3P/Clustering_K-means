import csv
from datetime import datetime

import numpy as np
import pandas as pd
import os

from sklearn.metrics import accuracy_score





dataset_path = '../resources/mnist_small_knn'

matrix =[]
with open(dataset_path+'/'+'train.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)
    matrix = np.array(data, dtype=int)

samples = matrix[:, 1:]
labels = matrix[:, 0]

print(samples)
print(labels)


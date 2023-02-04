import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd

#read the data from CSV file
data = pd.read_csv('filename')

#split training data and testing data
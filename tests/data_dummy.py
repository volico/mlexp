import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import KFold


class TrainerDummyData:
    def __init__(self):
        self.random_state = 54321
        np.random.seed(self.random_state)
        self.X = load_diabetes()['data']
        self.y = load_diabetes()['target']
        kf = KFold(n_splits=2, random_state=self.random_state, shuffle=True)
        self.cv_list = list(kf.split(self.X))

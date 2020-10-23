from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

def get_x_poly(time_period, degree):
    polynomial_features = PolynomialFeatures(degree = degree)
    return polynomial_features.fit_transform(time_period)

class PolynomialRegression:

    def __init__(self, degree):
        self.degree = degree
        self.model = None

    def train(self, time_period, state):
        self.model = LinearRegression()
        self.model.fit(get_x_poly(time_period, self.degree), state)

    def get_predictions(self, time_period):
        return np.round(self.model.predict(get_x_poly(time_period, self.degree)), 0).astype(np.int32)
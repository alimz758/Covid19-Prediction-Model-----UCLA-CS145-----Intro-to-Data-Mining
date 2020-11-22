import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import importlib

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score

from ..utils.prediction_model import PredictionModel
from ..utils.create_input_df import CreateDataframe

MIN_POLY_DEGREE = 1
MAX_POLY_DEGREE = 6


class PolynomialRegression(PredictionModel):
    def __init__(self):
        super(PolynomialRegression, self).__init__()
        self.model = LinearRegression()
        self.best_degree = None

    def train(self, predict_state, predict_field):
        self.model = LinearRegression()

        train_df = self.assign_train_df(predict_field)
        y_train = np.array(train_df[predict_state]).reshape(-1,)

        # Perform cross-validation
        validation_errors = []
        for i in range(MIN_POLY_DEGREE, MAX_POLY_DEGREE):
            poly = PolynomialFeatures(degree=i)
            x_transform = poly.fit_transform(self.x_train)

            validation_scores = cross_val_score(
                self.model, x_transform, y=y_train, scoring='neg_mean_squared_error', cv=10)
            validation_errors += [math.sqrt(- np.average(validation_scores))]

        # Find best polynomial degree
        validation_errors = np.array(validation_errors)
        self.best_degree = np.where(
            validation_errors == validation_errors.min())[0][0] + 1

        # Train and predict best model
        poly = PolynomialFeatures(degree=self.best_degree)
        x_transform = poly.fit_transform(self.x_train)
        self.model.fit(x_transform, y_train)

        # y_train_pred = self.model.predict(x_transform)
        # Draw scatter plot
        #self.scatter_plot(x_train, y_train, y_train_pred)

    def predict(self):
        poly = PolynomialFeatures(degree=self.best_degree)
        x_test_transform = poly.fit_transform(self.x_test)
        y_pred = self.model.predict(x_test_transform)
        self.y_pred = y_pred

        return np.round(y_pred, 0).astype(np.int32)
    
    def mape(self):
        return np.mean(np.abs((self.y_test - self.y_pred) / self.y_test)) * 100

def main():
    poly = PolynomialRegression()
    poly.train("Alabama", "Confirmed")
    prediction = poly.predict()
    print(prediction)


if __name__ == "__main__":
    main()

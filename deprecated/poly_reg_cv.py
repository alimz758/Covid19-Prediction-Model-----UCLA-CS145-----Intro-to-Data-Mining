# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import importlib

from sklearn import metrics
from sklearn.preprocessing import StandardScaler
# from create_final_df import CreateDataframe

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score

from create_input_df import CreateDataframe

# importlib.import_module('create_input_df', 'CreateDataframe')


class PolyRegCV(object):
    def polyReg(self, state='California', case_type='Confirmed'):

        # Initialize objects
        lr = LinearRegression()
        df = CreateDataframe()

        # Get dataframe
        p_df = df.get_final_df(case_type)

        # Initialize test and train inputs
        X_train = np.array([[num] for num in range(0, 142)])
        X_test = np.array([[num] for num in range(142, 168)])

        y_train = np.array(p_df[state]).reshape(-1,)

        # Perform cross-validation
        validation_errors = []
        for i in range(1, 6):
            poly = PolynomialFeatures(degree=i)
            X_transform = poly.fit_transform(X_train)

            validation_scores = cross_val_score(
                lr, X_transform, y=y_train, scoring='neg_mean_squared_error', cv=10)
            validation_errors += [math.sqrt(- np.average(validation_scores))]

        # Find best polynomial degree
        validation_errors = np.array(validation_errors)
        best_degree = np.where(validation_errors ==
                               validation_errors.min())[0][0] + 1
    #     print(validation_errors)
    #     print(best_degree)

        # Train and predict best model
        poly = PolynomialFeatures(degree=best_degree)
        X_transform = poly.fit_transform(X_train)
        lr.fit(X_transform, y_train)
        y_train_pred = lr.predict(X_transform)

    # Plot scatter
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #     ax.scatter(X_train, y_train)
    #     ax.plot(X_train, y_train_pred, color='r')

        X_test_transform = poly.fit_transform(X_test)
        y_pred = lr.predict(X_test_transform)

        return y_pred


def main():
    poly = PolyRegCV()
    prediction = poly.polyReg("Alabama", "Confirmed")
    # prediction = poly.predict()
    print(prediction)


if __name__ == "__main__":
    main()

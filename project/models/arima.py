
import numpy as np
import itertools
import pandas as pd
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

from ..utils.prediction_model import PredictionModel
from ..utils.create_input_df import CreateDataframe


FUTURE_DAYS = 26

class ARIMA(PredictionModel):
    def __init__(self):
        super(ARIMA, self).__init__()
        # Define the p, d and q parameters to take any value between 0 and 2
        self.p = self.d = self.q = range(0, 2)
        # Generate all different combinations of p, q and q triplets
        self.pdq = list(itertools.product(self.p, self.d, self.q))
        # Generate all different combinations of seasonal p, q and q triplets
        self.seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(self.p, self.d, self.q))]
        self.best_aic = float('inf')
        self.best_param = None
        self.best_param_seasonal = None
        self.model = None
        self.results = None

    def train(self, predict_state, predict_field):
        train_df = self.assign_train_df(predict_field)
        for param in self.pdq:
            for param_seasonal in self.seasonal_pdq:
                try:
                    mod = sm.tsa.statespace.SARIMAX(train_df[predict_state],
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

                    results = mod.fit(disp=False)
                    if results.aic < self.best_aic:
                        self.best_aic = results.aic
                        self.best_param = param
                        self.best_param_seasonal = param_seasonal
                except:
                    continue
        self.model = sm.tsa.statespace.SARIMAX(train_df[predict_state],
                                order=self.best_param,
                                seasonal_order=self.best_param_seasonal,
                                enforce_stationarity=False,
                                enforce_invertibility=False)

        self.results = self.model.fit(disp=False)

    def predict(self):
        pred = self.results.get_forecast(steps=FUTURE_DAYS)
        return np.round(pred.predicted_mean, 0).astype(np.int32).array



def main():
    arima = ARIMA()
    arima.train("Wyoming", "Confirmed")
    prediction = arima.predict()
    print(prediction)


if __name__ == "__main__":
    main()

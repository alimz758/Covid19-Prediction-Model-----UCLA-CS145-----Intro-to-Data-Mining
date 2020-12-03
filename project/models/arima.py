
import numpy as np
import itertools
import pandas as pd

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

from ..utils.prediction_model import PredictionModel
from ..utils.create_input_df import CreateDataframe

#first round
# FUTURE_DAYS = 26

#second round
# 12-07-2020 to 12-13-2020 is 7 days. 
# 11-22-2020 to 12-07-2020 is 16 days so ignore the first 15 days in prediction below.
# FUTURE_DAYS = 22

# FUTURE_DAYS = 7

# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(X, arima_order):
    # prepare training dataset
    train_size = int(len(X.values) * 0.66)
    train, test = X.values[0:train_size], X.values[train_size:]
    history = [x for x in train]  
    predictions = list()
    try:
        for t in range(len(test)):
            model = ARIMA(history, order=arima_order)
            model_fit = model.fit()
            yhat = model_fit.forecast()[0]
            predictions.append(yhat) 
            history.append(test[t])        
    except:
        pass
    
    if len(test)>len(predictions):
        error = mean_squared_error(test[:len(predictions)], predictions)
    else:
        error = mean_squared_error(test, predictions[:len(test)])
    return error

class ARIMA_MODEL(PredictionModel):
    def __init__(self, whichRound):
        super(ARIMA_MODEL, self).__init__(whichRound)
        # Define the p, d and q parameters to take any value between 0 and 2
        self.p = self.d = self.q = range(0, 2)
        # Generate all different combinations of p, q and q triplets
        self.pdq = list(itertools.product(self.p, self.d, self.q))
        self.best_param = None
        self.best_mse = float('inf')
        self.model = None
        self.results = None

    def train(self, predict_state, predict_field):
        self.train_df = self.assign_train_df(predict_field)
        self.state = predict_state
        for param in self.pdq:
            try:
                mse = evaluate_arima_model(self.train_df[predict_state], param)
                if mse < self.best_mse:
                    self.best_mse, self.best_param = mse, param
            except:
                continue 
        self.model = ARIMA(self.train_df[predict_state],
                            order=self.best_param)

        self.results = self.model.fit()

# future_days, pass in either round 1 or round 2 prediction range
    def predict(self, future_days):
        pred = self.results.predict(
                start=len(self.train_df[self.state]), 
                end=len(self.train_df[self.state]) + future_days - 1, 
                dynamic=False)
        return np.round(pred, 0).astype(np.int32).array



def main():
    arima = ARIMA_MODEL()
    arima.train("Wyoming", "Confirmed")
    prediction = arima.predict()
    print(prediction)


if __name__ == "__main__":
    main()

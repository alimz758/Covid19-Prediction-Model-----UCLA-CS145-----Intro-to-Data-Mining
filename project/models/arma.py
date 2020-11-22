
import numpy as np
import pandas as pd
<<<<<<< HEAD
from statsmodels.tsa.arima_model import ARMA
=======
import itertools
from statsmodels.tsa.arima_model import ARMA
from sklearn.metrics import mean_squared_error
>>>>>>> new models trial- ARIMA doesn't converge

from ..utils.prediction_model import PredictionModel
from ..utils.create_input_df import CreateDataframe


FUTURE_DAYS = 26

<<<<<<< HEAD
class ARMA_MODEL(PredictionModel):
    def __init__(self):
        super(ARMA_MODEL, self).__init__()
=======
def evaluate_arima_model(X, arima_order):
    # prepare training dataset
    train_size = int(len(X.values) * 0.66)
    train, test = X.values[0:train_size], X.values[train_size:]
    history = [x for x in train]  
    predictions = list()
    try:
        for t in range(len(test)):
            model = ARMA(history, order=arima_order)
            model_fit = model.fit(disp=False, transparams=False)
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

class ARMA_MODEL(PredictionModel):
    def __init__(self):
        super(ARMA_MODEL, self).__init__()
        # Define the p, d and q parameters to take any value between 0 and 2
        self.p = self.q = range(0, 3)
        # Generate all different combinations of p, q and q triplets
        self.pq = list(itertools.product(self.p, self.q))
        self.best_param = None
        self.best_mse = float('inf')
>>>>>>> new models trial- ARIMA doesn't converge
        self.model = None
        self.model_fitted = None
        self.train_df = None
        
    def train(self, predict_state, predict_field):
        self.train_df = self.assign_train_df(predict_field)
        self.state = predict_state
<<<<<<< HEAD
        self.model = ARMA(self.train_df[predict_state], order=(2, 1))
        self.model_fitted = self.model.fit(disp=False)

    def predict(self):
        pred = self.model_fitted.predict(
=======
        for param in self.pq:
            try:
                mse = evaluate_arima_model(self.train_df[predict_state], param)
                if mse < self.best_mse:
                    self.best_mse, self.best_param = mse, param
            except:
                continue 
        self.model = ARMA(self.train_df[predict_state],
                            order=self.best_param)

        self.results = self.model.fit(disp=False, transparams=False)

    def predict(self):
        pred = self.results.predict(
>>>>>>> new models trial- ARIMA doesn't converge
                start=len(self.train_df[self.state]), 
                end=len(self.train_df[self.state]) + FUTURE_DAYS - 1, 
                dynamic=False)
        return np.round(pred, 0).astype(np.int32).array



def main():
    arma = ARMA_MODEL()
    arma.train("Wyoming", "Confirmed")
    prediction = arma.predict()
    print(prediction)


if __name__ == "__main__":
    main()

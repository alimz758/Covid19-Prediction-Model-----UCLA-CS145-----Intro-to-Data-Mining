
import numpy as np
import pandas as pd
from statsmodels.tsa.arima_model import ARMA

from ..utils.prediction_model import PredictionModel
from ..utils.create_input_df import CreateDataframe


FUTURE_DAYS = 26

class MovingAverage(PredictionModel):
    def __init__(self):
        super(MovingAverage, self).__init__()
        self.model = None
        self.model_fitted = None
        self.train_df = None
        
    def train(self, predict_state, predict_field):
        self.train_df = self.assign_train_df(predict_field)
        self.state = predict_state
        self.model = ARMA(self.train_df[predict_state],  order=(0,1, 10))
        self.model_fitted = self.model.fit(disp=False, transparams=False)

    def predict(self):
        pred = self.model_fitted.predict(
                start=len(self.train_df[self.state]), 
                end=len(self.train_df[self.state]) + FUTURE_DAYS - 1, 
                dynamic=False)
        return np.round(pred, 0).astype(np.int32).array



def main():
    ma = MovingAverage()
    ma.train("Wyoming", "Confirmed")
    prediction = ma.predict()
    print(prediction)


if __name__ == "__main__":
    main()

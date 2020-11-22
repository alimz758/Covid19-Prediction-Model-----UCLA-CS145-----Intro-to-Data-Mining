
import numpy as np
import pandas as pd
from statsmodels.tsa.arima_model import ARMA

from ..utils.prediction_model import PredictionModel
from ..utils.create_input_df import CreateDataframe


FUTURE_DAYS = 26

class ARMA_MODEL(PredictionModel):
    def __init__(self):
        super(ARMA_MODEL, self).__init__()
        self.model = None
        self.model_fitted = None
        self.train_df = None
        
    def train(self, predict_state, predict_field):
        self.train_df = self.assign_train_df(predict_field)
        self.state = predict_state
        self.model = ARMA(self.train_df[predict_state], order=(2, 1))
        self.model_fitted = self.model.fit(disp=False)

    def predict(self):
        pred = self.model_fitted.predict(
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

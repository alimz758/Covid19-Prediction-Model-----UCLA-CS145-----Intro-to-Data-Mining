
import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AR

from ..utils.prediction_model import PredictionModel
from ..utils.create_input_df import CreateDataframe


FUTURE_DAYS = 26

class AutoRegression(PredictionModel):
    def __init__(self):
        super(AutoRegression, self).__init__()
        self.model = None
        self.model_fitted = None
        self.train_df = None
        
    def train(self, predict_state, predict_field):
        self.train_df = self.assign_train_df(predict_field)
        self.state = predict_state
        self.model = AR(self.train_df[predict_state])
        self.model_fitted = self.model.fit(maxlag=6)

    def predict(self):
        pred = self.model_fitted.predict(
                start=len(self.train_df[self.state]), 
                end=len(self.train_df[self.state]) + FUTURE_DAYS - 1, 
                dynamic=False)
        return np.round(pred, 0).astype(np.int32).array



def main():
    ar = AutoRegression()
    ar.train("Wyoming", "Confirmed")
    prediction = ar.predict()
    print(prediction)


if __name__ == "__main__":
    main()

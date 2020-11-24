from .create_input_df import CreateDataframe
import numpy as np

PREDICTION_DAYS_COUNT = 26


class PredictionModel:
    def __init__(self):
        self.dataFrameFactory = CreateDataframe()
        self.train_data_confirmed = self.dataFrameFactory.get_final_df(
            "Confirmed")[:142]
        self.test_data_confirmed = self.dataFrameFactory.get_final_df(
            "Confirmed")[142:]
        self.train_data_death = self.dataFrameFactory.get_final_df("Deaths")[:142]
        self.test_data_death = self.dataFrameFactory.get_final_df("Deaths")[142:]
        
        assert(len(np.array(self.train_data_confirmed["Days"])) == len(
            np.array(self.train_data_death["Days"])))

        # training input: array of date index (i.e 1,2,...142)
        self.x_train = np.array(
            self.train_data_confirmed["Days"]).reshape(-1, 1)[:142]
        # testing input: array of date index, following the training input (i.e 142,143,...167)
        self.x_test = np.array(
            self.test_data_confirmed["Days"]).reshape(-1, 1)
        

    def assign_train_df(self, predict_field):
        if predict_field == "Confirmed":
            self.y_test = self.test_data_confirmed
            return self.train_data_confirmed
        elif predict_field == "Deaths":
            self.y_test = self.test_data_death
            return self.train_data_death
    

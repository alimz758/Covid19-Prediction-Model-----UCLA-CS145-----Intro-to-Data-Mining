from create_input_df import CreateDataframe
import numpy as np

PREDICTION_DAYS_COUNT = 26


class PredictionModel:
    def __init__(self):
        self.dataFrameFactory = CreateDataframe()
        self.train_data_confirmed = self.dataFrameFactory.get_final_df(
            "Confirmed")
        self.train_data_death = self.dataFrameFactory.get_final_df("Deaths")
        assert(len(np.array(self.train_data_confirmed["Days"])) == len(
            np.array(self.train_data_death["Days"])))

        # training input: array of date index (i.e 1,2,...142)
        self.x_train = np.array(
            self.train_data_confirmed["Days"]).reshape(-1, 1)
        # testing input: array of date index, following the training input (i.e 142,143,...167)
        self.x_test = np.array(range(
            len(self.x_train), len(self.x_train) + PREDICTION_DAYS_COUNT)).reshape(-1, 1)

    def assign_train_df(self, predict_field):
        if predict_field == "Confirmed":
            return self.train_data_confirmed
        elif predict_field == "Deaths":
            return self.train_data_death

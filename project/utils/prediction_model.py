from .create_input_df import CreateDataframe
import numpy as np

# round 1
# PREDICTION_DAYS_COUNT = 26

# round 2
# predict 22 days but trim off first 15 days because we're predicting some time with a gap ahead of last trained date
# PREDICTION_DAYS_COUNT = 22
PREDICTION_DAYS_COUNT = 7

# ls | wc -l to count number of files: 225 which are number of days of data
# we are trainign up to 

class PredictionModel:
    def __init__(self):
        self.dataFrameFactory = CreateDataframe()
        # round 1 prediction
        # self.train_data_confirmed = self.dataFrameFactory.get_final_df(
        #     "Confirmed")[:142]
        # self.test_data_confirmed = self.dataFrameFactory.get_final_df(
        #     "Confirmed")[142:]
        # self.train_data_death = self.dataFrameFactory.get_final_df("Deaths")[:142]
        # self.test_data_death = self.dataFrameFactory.get_final_df("Deaths")[142:]

        #round 2 prediction
        # self.train_data_confirmed = self.dataFrameFactory.get_final_df(
        #     "Confirmed")[:225]
        # self.test_data_confirmed = self.dataFrameFactory.get_final_df(
        #     "Confirmed")[204:]
        # self.train_data_death = self.dataFrameFactory.get_final_df("Deaths")[:225]
        # self.test_data_death = self.dataFrameFactory.get_final_df("Deaths")[204:]

        self.train_data_confirmed = self.dataFrameFactory.get_final_df(
            "Confirmed")[:218]
        self.test_data_confirmed = self.dataFrameFactory.get_final_df(
            "Confirmed")[218:]
        self.train_data_death = self.dataFrameFactory.get_final_df("Deaths")[:218]
        self.test_data_death = self.dataFrameFactory.get_final_df("Deaths")[218:]
        
        assert(len(np.array(self.train_data_confirmed["Days"])) == len(
            np.array(self.train_data_death["Days"])))

        # round 1
        # training input: array of date index (i.e 1,2,...142)
        # self.x_train = np.array(
        #     self.train_data_confirmed["Days"]).reshape(-1, 1)[:142]

        # round 2
        self.x_train = np.array(
            self.train_data_confirmed["Days"]).reshape(-1, 1)[:218]
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
    

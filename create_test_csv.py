from analysis import PREDICTION_DAYS_COUNT
import numpy as np
from project.utils.create_input_df import CreateDataframe
import pandas as pd

# round 1
# PREDICTION_DAYS_COUNT = 26
# round 2
# PREDICTION_DAYS_COUNT = 22
PREDICTION_DAYS_COUNT = 7
FILE_NAME = "Test.csv"
STATES_COUNT = 50
STATE_CSV_FILE_PATH = "./project/data/daily_report_per_states/states/states.csv"

def get_forecast_id(date_day, state_id):
    return state_id + STATES_COUNT * date_day

def get_test_df(self, state_id, attr):
    if attr == "Confirmed":
        return self.test_data_confirmed[self.US_STATES[state_id]]
    else:
        return self.test_data_death[self.US_STATES[state_id]]
    


class CreateTestCSV(object):
    def __init__(self):
        self.dataFrameFactory = CreateDataframe()
        # round 1
        # self.test_data_confirmed = self.dataFrameFactory.get_final_df("Confirmed")[142:]
        # self.test_data_death = self.dataFrameFactory.get_final_df("Deaths")[142:]
        # testing input: array of date index, following the training input (i.e 142,143,...167)

        self.test_data_confirmed = self.dataFrameFactory.get_final_df("Confirmed")[218:]
        self.test_data_death = self.dataFrameFactory.get_final_df("Deaths")[218:]
        #testing input: array of date index, following the training input (i.e 204,204,...225)
        # round 1
        # self.days = np.array(
        #     self.test_data_confirmed["Days"]).reshape(-1, 1)[142:]

        #round 2
        self.days = np.array(
            self.test_data_confirmed["Days"]).reshape(-1, 1)[218:]
        states_file = STATE_CSV_FILE_PATH
        states = pd.read_csv(states_file, engine="python")
        self.US_STATES = []
        for index, row in states.iterrows():
            self.US_STATES.append(row.loc['State'])


    def generate(self):
        deaths = [None] * STATES_COUNT
        confirmed = [None] * STATES_COUNT
        res = []

        # get predicted values for each state
        for state_id in range(STATES_COUNT):
            confirmed[state_id] = get_test_df(self, state_id, "Confirmed")
            deaths[state_id] = get_test_df(self, state_id, "Deaths")

# round 1
        # for day in range(142, 142 + PREDICTION_DAYS_COUNT):
        #     for state_id in range(STATES_COUNT):
        #         forcast_id = get_forecast_id(day-142, state_id)
        #         res.append([forcast_id, confirmed[state_id]
        #                     [day], deaths[state_id][day]])
# round 2
        print('deaths shape', deaths)
        for day in range(218, 218 + PREDICTION_DAYS_COUNT):
            for state_id in range(STATES_COUNT):
                forcast_id = get_forecast_id(day-218, state_id)
                print('forecast id', forcast_id)
                print('day', day)
                res.append([forcast_id, confirmed[state_id]
                            [day], deaths[state_id][day]])
        

        return res
    
    def write_file(self, data):
        file = open(FILE_NAME, "w")
        file.truncate()
        file.write("ForecastID,Confirmed,Deaths\n")
        for row in data:
            line = str(row[0]) + "," + str(row[1]) + \
                "," + str(row[2]) + "\n"
            file.write(line)
    
    
def main():
    csv = CreateTestCSV()
    output = csv.generate()
    csv.write_file(output)


if __name__ == "__main__":
    main()

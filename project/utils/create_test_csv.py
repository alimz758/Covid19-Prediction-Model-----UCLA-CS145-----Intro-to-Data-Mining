import numpy as np
from .project.utils.create_input_df import CreateDataframe

PREDICTION_DAYS_COUNT = 26
FILE_NAME = "Test.csv"
STATES_COUNT = 50
US_STATES = []


def get_forecast_id(date_day, state_id):
    return state_id + STATES_COUNT * date_day

def get_test_df(self, state_id, attr):
    if attr == "Confirmed":
        return self.test_data_confirmed[state_id]
    else:
        return self.test_data_death[state_id]
    


class CreateTestCSV(object):
    def __init__(self):
        self.dataFrameFactory = CreateDataframe()
        self.test_data_confirmed = self.dataFrameFactory.get_final_df("Confirmed")[142:]
        
        self.test_data_death = self.dataFrameFactory.get_final_df("Deaths")[142:]
        
        # testing input: array of date index, following the training input (i.e 142,143,...167)
        self.days = np.array(
            self.test_data_confirmed["Days"]).reshape(-1, 1)[142:]

    def generate(self):
        deaths = [None] * STATES_COUNT
        confirmed = [None] * STATES_COUNT
        res = []

        # get predicted values for each state
        for state_id in range(STATES_COUNT):
            confirmed[state_id] = get_test_df(self, state_id, "Confirmed")
            deaths[state_id] = get_test_df(self, state_id, "Deaths")

        for day in range(PREDICTION_DAYS_COUNT):
            for state_id in range(STATES_COUNT):
                forcast_id = get_forecast_id(day, state_id)
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

    output = csv.predict()
    csv.write_file(output)


if __name__ == "__main__":
    main()

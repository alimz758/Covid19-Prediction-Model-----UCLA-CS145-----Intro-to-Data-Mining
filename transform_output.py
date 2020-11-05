import csv
import pandas as pd


# from sys import argv
# import os.path


# def file_exists(path):
#     if(os.path.isfile(path)):
#         return True
#     return False

# sort state order, and index from 0 to 49
# for each state
US_STATES = []
NUMBER_OF_DAYS = 26
STATES_COUNT = 50
SUBMISSION_FILE_NAME = "submission"
STATE_CSV_FILE_PATH = './data/daily_report_per_states/states/states.csv'


def init():
    states_file = STATE_CSV_FILE_PATH
    states = pd.read_csv(states_file, engine="python")
    for index, row in states.iterrows():
        US_STATES.append(row.loc['State'])

# date_day: date of the month under evaluation
# state_id: index of the state in the alphabetically sorted states array (ex: Alabama: 0)


def get_forecast_id(date_day, state_id):
    return date_day - 1 + STATES_COUNT * state_id

# model_type: either "NN" or "PR"


def predict(model_type):
    prediction_model = None
    predicted_deaths_values = []
    predicted_confirmed_values = []
    res = []

    if model_type == "NN":
        prediction_model = get_NN_prediction
    elif model_type == "PR":
        prediction_model = get_PR_prediction
    else:
        raise ValueError("Model not recognized")

    # get predicted values for each state
    for state_id in range(STATES_COUNT):
        print("Training & predicting for ", US_STATES[state_id])
        predicted_deaths_values[states_id] = prediction_model(
            state_id, "Deaths")
        predicted_confirmed_values[state_id] = prediction_model(
            state_id, "Confirmed")

    for day in range(NUMBER_OF_DAYS):
        for state_id in range(STATES_COUNT):
            forcast_id = get_forecast_id(day, state_id)
            # state = US_STATES[state_id]
            res[forecast_id] = [forcast_id, predicted_confirmed_values[state_id]
                                [day], predicted_deaths_values[state_id][day]]

    return res


def get_PR_prediction(state_id, prediction_type):
    return


def get_NN_prediction():
    return

# prediction_values: 2D array, [<String>forecast_id][<Array>(forecast_id, confirmed_values, death_values)]


def write_file(prediction_values):
    file = open(SUBMISSION_FILE_NAME, "w")
    file.truncate()

    for row in prediction_values:
        line = row[0] + "," + row[1][1] + "," + row[1][2] + "\n"
        file.write(line)


def main():
    init()
    output = predict("NN")
    write_file(output)
    # output = predict("PR")

    # input_file_name = argv[1]
    # if not file_exists(input_file_name):
    #     raise ValueError("File with filename not found")


if __name__ == "__main__":
    main()

import sys
import csv
import pandas as pd

from polynomial_regression import PolynomialRegression
from neural_network import NeuralNetwork
from arima import ARIMA
from auto_regression import AutoRegression

US_STATES = []
NUMBER_OF_DAYS = 26
STATES_COUNT = 50
SUBMISSION_FILE_NAME = "Team14.csv"
STATE_CSV_FILE_PATH = './data/daily_report_per_states/states/states.csv'
ACCEPTED_MODEL_TYPES = ["NN", "PR", "ARIMA", "AR"]


def init():
    states_file = STATE_CSV_FILE_PATH
    states = pd.read_csv(states_file, engine="python")
    for index, row in states.iterrows():
        US_STATES.append(row.loc['State'])

# date_day: date of the month under evaluation
# state_id: index of the state in the alphabetically sorted states array (ex: Alabama: 0)


def get_forecast_id(date_day, state_id):
    return state_id + STATES_COUNT * date_day

# model_type: either "NN" or "PR"


def predict(model_type):
    prediction_model = None
    predicted_deaths_values = [None] * STATES_COUNT
    predicted_confirmed_values = [None] * STATES_COUNT
    res = []

    if model_type == "NN":
        prediction_model = get_NN_prediction
    elif model_type == "PR":
        prediction_model = get_PR_prediction
    elif model_type == "ARIMA":
        prediction_model = get_ARIMA_prediction
    elif model_type == "AR":
        prediction_model = get_AR_prediction
    else:
        raise ValueError("Model not recognized")

    # get predicted values for each state
    for state_id in range(STATES_COUNT):
        print("Training & predicting for ", US_STATES[state_id])
        predicted_deaths_values[state_id] = prediction_model(
            state_id, "Deaths")
        predicted_confirmed_values[state_id] = prediction_model(
            state_id, "Confirmed")

    for day in range(NUMBER_OF_DAYS):
        for state_id in range(STATES_COUNT):
            forcast_id = get_forecast_id(day, state_id)
            res.append([forcast_id, predicted_confirmed_values[state_id]
                        [day], predicted_deaths_values[state_id][day]])

    return res


def get_PR_prediction(state_id, prediction_type):
    pr_model = PolynomialRegression()
    pr_model.train(US_STATES[state_id], prediction_type)
    return pr_model.predict()


def get_NN_prediction(state_id, prediction_type):
    pr_model = NeuralNetwork()
    pr_model.train(US_STATES[state_id], prediction_type)
    return pr_model.predict()

def get_ARIMA_prediction(state_id, prediction_type):
    arima_model = ARIMA()
    arima_model.train(US_STATES[state_id], prediction_type)
    return arima_model.predict()

def get_AR_prediction(state_id, prediction_type):
    ar_model = AutoRegression()
    ar_model.train(US_STATES[state_id], prediction_type)
    return ar_model.predict()



# prediction_values: 2D array, [<String>forecast_id][<Array>(forecast_id, confirmed_values, death_values)]


def write_file(prediction_values):
    file = open(SUBMISSION_FILE_NAME, "w")
    file.truncate()

    for row in prediction_values:
        line = str(row[0]) + "," + str(row[1]) + \
            "," + str(row[2]) + "\n"
        file.write(line)


def main():
    init()
    model_type = sys.argv[1]
    if not model_type in ACCEPTED_MODEL_TYPES:
        raise ValueError("Only " + str(ACCEPTED_MODEL_TYPES) + " accepted")

    output = predict(model_type)
    write_file(output)


if __name__ == "__main__":
    main()

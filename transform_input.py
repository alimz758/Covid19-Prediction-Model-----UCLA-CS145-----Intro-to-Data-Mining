import pandas as pd
import numpy as np
import os.path

DAYS_COUNT = {4: 30, 5: 31, 6: 30, 7: 31, 8: 31}
PER_STATE_DATA_PATH = './data/daily_report_per_states/'


def file_exists(path):
    if(os.path.isfile(path)):
        return True
    return False


def truncate_all_state_files():
    states = pd.read_csv(
        './data/daily_report_per_states/states.csv', engine="python")
    for index, row in states.iterrows():
        state = row.loc['State']
        path = PER_STATE_DATA_PATH + state + ".csv"
        if(not file_exists(path)):
            open(path, "x")
        file = open(path, "w")
        file.truncate()


def day_str(date):
    if (date < 10):
        return "0" + str(date)
    return str(date)

# partition input data by state


def partition_state(data):
    for index, row in data.iterrows():
        state_csv_file = PER_STATE_DATA_PATH + \
            row.loc["Province_State"] + ".csv"
        if (not file_exists(state_csv_file)):
            continue
        for
        print(row.values)

# append csv files


def main():
    truncate_all_state_files()

    for month in DAYS_COUNT:
        for day in range(1, DAYS_COUNT[month]+1):
            date = "0" + str(month) + "-" + day_str(day) + "-2020.csv"
            path = './data/daily_report/' + date
            if(not file_exists(path)):
                continue
            daily_report_data = pd.read_csv(path, engine="python")
            partition_state(daily_report_data)
            # add methods here


if __name__ == "__main__":
    main()

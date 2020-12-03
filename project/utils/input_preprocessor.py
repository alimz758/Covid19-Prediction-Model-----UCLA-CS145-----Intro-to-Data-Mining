import pandas as pd
import numpy as np
import math
import os.path


# round 1
# DAYS_COUNT = {4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 26}

#round 2
# set november to 22 because we only have data up to the 22nd, hypothetically at the time
DAYS_COUNT = {4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12:2}
PER_STATE_DATA_PATH = '../data/daily_report_per_states/'
KEPT_FIELDS = ["Confirmed", "Deaths", "Recovered", "Active", "FIPS", "Incident_Rate", "People_Tested",
               "People_Hospitalized", "Mortality_Rate", "UID", "ISO3", "Testing_Rate", "Hospitalization_Rate"]


def file_exists(path):
    if(os.path.isfile(path)):
        return True
    return False


def reset_all_state_files():
    states_file = PER_STATE_DATA_PATH + 'states/states.csv'
    states = pd.read_csv(states_file, engine="python")
    for index, row in states.iterrows():
        state = row.loc['State']
        path = PER_STATE_DATA_PATH + state + ".csv"
        if(not file_exists(path)):
            open(path, "x")
        file = open(path, "w")
        file.truncate()
        init_row = "Date"
        for field in KEPT_FIELDS:
            init_row += "," + field
        init_row += "\n"
        file.write(init_row)


def day_str(date):
    if (date < 10):
        return "0" + str(date)
    return str(date)

# partition input data by state


def partition_state(data, date):
    for index, row in data.iterrows():
        state_csv_file = PER_STATE_DATA_PATH + \
            row.loc["Province_State"] + ".csv"
        if (not file_exists(state_csv_file)):
            continue
        csv_row = date
        for field in row.values[5:]:
            if (isinstance(field, str)):
                csv_row = csv_row + ',' + field
            elif (math.isnan(field)):
                csv_row = csv_row + ','
            else:
                csv_row = csv_row + ',' + str(field)
        csv_row = csv_row + "\n"
        with open(state_csv_file, 'a') as fd:
            fd.write(csv_row)
        # append csv files


def main():
    reset_all_state_files()
    # daily_report_data = pd.read_csv('../data/daily_report/11-22-2020.csv', engine="python")
    # print(daily_report_data)
    for month in DAYS_COUNT:
        # print('month', month)
        for day in range(1, DAYS_COUNT[month]+1):
            # print('month day', month, day)
            date = ""
            if month >= 10:
              date = str(month) + "-" + day_str(day)
            else:
              date = "0" + str(month) + "-" + day_str(day)
            filename = date + "-2020.csv"
            file_path = '../data/daily_report/' + filename
            if(not file_exists(file_path)):
                print('skipped', file_path)
                continue
            daily_report_data = pd.read_csv(file_path, engine="python")
            # print(daily_report_data)
            partition_state(daily_report_data, date)
            # add methods here


if __name__ == "__main__":
    main()

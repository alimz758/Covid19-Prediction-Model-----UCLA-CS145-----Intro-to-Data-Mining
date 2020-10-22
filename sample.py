import pandas as pd
import numpy as np


def day_str(date):
    if (date < 10):
        return "0" + str(date)
    return str(date)

# def partition_state(daily_csv, date):


days_count = {4: 30, 5: 31, 6: 30, 7: 31, 8: 31}
for month in days_count:
    for day in range(1, days_count[month]+1):
        data_path = "0" + str(month) + "-" + day_str(day) + "-2020.csv"
        print("trying: ", './data/daily_report/' + data_path)
        try:
            daily_report_04_12_df = pd.read_csv(
                './data/daily_report/' + data_path, engine="python")
            print("imported: " + data_path)
        except FileNotFoundError:
            pass

import sys
import csv
import pandas as pd
import numpy as np

TEST_PATH = "./Test.csv"
TRAIN_PATH = "./Team14.csv"

def get_mape(test, train):
    return np.mean(np.abs((test - train)/test)) * 100

def main():
    test_df = pd.read_csv(TEST_PATH)
    train_df = pd.read_csv(TRAIN_PATH)
    mape_confirmed = get_mape(test_df['Confirmed'], train_df['Confirmed'])
    mape_deaths = get_mape(test_df['Deaths'], train_df['Deaths'])
    print("MAPE of Confirmed {}".format(mape_confirmed))
    print("MAPE of Deaths {}".format(mape_deaths))
    print('Score is: {}'.format((mape_confirmed+mape_deaths)/2))


if __name__ == "__main__":
    main()

from statsmodels.tsa.stattools import adfuller
from project.utils.create_input_df import CreateDataframe
from scipy import stats
from scipy.stats import normaltest
import statsmodels.api as sm
import seaborn as sns
# from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def test_stationarity(timeseries, window = 12, cutoff = 0.01):

    #Determing rolling statistics
    rolmean = timeseries.rolling(window).mean()
    rolstd = timeseries.rolling(window).std()

    #Plot rolling statistics:
    fig = plt.figure(figsize=(12, 8))
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC', maxlag = 20 )
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    pvalue = dftest[1]
    if pvalue < cutoff:
        print('p-value = %.4f. The series is likely stationary.' % pvalue)
    else:
        print('p-value = %.4f. The series is likely non-stationary.' % pvalue)
    
    print(dfoutput)


PREDICTION_DAYS_COUNT = 26
# FILE_NAME = "Test.csv"
FILE_NAME = "analysis.csv"
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
        self.test_data_confirmed = self.dataFrameFactory.get_final_df("Confirmed")[142:]
        self.test_data_death = self.dataFrameFactory.get_final_df("Deaths")[142:]
        # testing input: array of date index, following the training input (i.e 142,143,...167)
        self.days = np.array(
            self.test_data_confirmed["Days"]).reshape(-1, 1)[142:]
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
            # confirmed[state_id] = get_test_df(self, state_id, "Confirmed")
            deaths[state_id] = get_test_df(self, state_id, "Deaths")

        for day in range(142, 142 + PREDICTION_DAYS_COUNT):
            for state_id in range(STATES_COUNT):
                forcast_id = get_forecast_id(day-142, state_id)
                # res.append([forcast_id, confirmed[state_id]
                            # [day], deaths[state_id][day]])
                res.append([forcast_id, deaths[state_id][day]])

        return res
    
    def write_file(self, data):
        file = open(FILE_NAME, "w")
        file.truncate()
        file.write("ForecastID,Deaths\n")
        for row in data:
            # line = str(row[0]) + "," + str(row[1]) + \
            #     "," + str(row[2]) + "\n"
            line = str(row[0]) + "," + str(row[1]) + "\n"
            file.write(line)
    
    
def main():
    csv = CreateTestCSV()
    output = csv.generate()
    csv.write_file(output)
    # confirmed_df = pd.read_csv("analysis.csv")
    death_df = pd.read_csv("analysis.csv")
    arima_mod6 = sm.tsa.ARIMA(death_df['Deaths'], (27,1,0)).fit(disp=False)
    print(arima_mod6.summary())


    resid = arima_mod6.resid
    print(normaltest(resid))
    # returns a 2-tuple of the chi-squared statistic, and the associated p-value. the p-value is very small, meaning
    # the residual is not a normal distribution

    fig = plt.figure(figsize=(12,8))
    ax0 = fig.add_subplot(111)

    sns.distplot(resid ,fit = stats.norm, ax = ax0) # need to import scipy.stats

    # Get the fitted parameters used by the function
    (mu, sigma) = stats.norm.fit(resid)

    #Now plot the distribution using 
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
    plt.ylabel('Frequency')
    plt.title('Residual distribution')


    # ACF and PACF
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(arima_mod6.resid, lags=40, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(arima_mod6.resid, lags=40, ax=ax2)
    plt.show()
    # fig = plt.figure(figsize=(12,8))
    # ax1 = fig.add_subplot(211)
    # fig = sm.graphics.tsa.plot_acf(death_df['Deaths'], lags=40, ax=ax1) # 
    # ax2 = fig.add_subplot(212)
    # fig = sm.graphics.tsa.plot_pacf(death_df['Deaths'], lags=40, ax=ax2)# , lags=40
    # plt.show()
    # print(confirmed_df)
    # print(output)
    # test_stationarity(death_df['Deaths'])
    # first_diff = death_df['Deaths'] - death_df['Deaths'].shift(1)
    # first_diff = first_diff.dropna(inplace = False)
    # test_stationarity(first_diff, window = 12)


if __name__ == "__main__":
    main()

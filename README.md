# Covid19-Prediction-Model-----UCLA-CS145-----Intro-to-Data-Mining

Course project for UCLA CS145, Introduction to Data Mining

## Running the Model

The main driver script is `run.py`. It takes in a single argument, the ML model type: [NN, PR, AR, ARIMA, ARMA, MA, SARIMA]

## Models used for prediction:

> PR: Polynomial Regression
> NN: Neural Network
> AR: Auto Regression
> MA: Mean Average
> ARIMA
> ARMA
> SARIMA

### ex)

> py run.py NN

This will generate a result csv file, matching the Kaggle submission format. To change any configurations, refer to the constant variables declared in run.py, polynomial_regression.py, neural_network.py, or prediction_model.py (superclass of all prediction models).

## Initializing Input Data

### Partitioning daily report data by states

To transform input data, run:

> python transform_input.py

It will then create a csv file for each states, each containing its state's daily report. Miscellaneous states from the input data set are ignored

**NOTE** Each time this script is ran, all the `<state>.csv` files are truncated an refilled from the daily report files.

### Data format (copied from https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data)

[USA daily state reports (csse_covid_19_daily_reports_us)](https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_daily_reports_us)

This table contains an aggregation of each USA State level data.


### Create the Test.csv

To create the test.csv file, run: 

> python create_test_csv.py

### Get MAPE

To get MAPE of the prediction vs truth data, run:

> python mape.py

### File naming convention

MM-DD-YYYY.csv in UTC.

### Field description

- <b>Province_State</b> - The name of the State within the USA.
- <b>Country_Region</b> - The name of the Country (US).
- <b>Last_Update</b> - The most recent date the file was pushed.
- <b>Lat</b> - Latitude.
- <b>Long\_</b> - Longitude.
- <b>Confirmed</b> - Aggregated case count for the state.
- <b>Deaths</b> - Aggregated death toll for the state.
- <b>Recovered</b> - Aggregated Recovered case count for the state.
- <b>Active</b> - Aggregated confirmed cases that have not been resolved (Active cases = total cases - total recovered - total deaths).
- <b>FIPS</b> - Federal Information Processing Standards code that uniquely identifies counties within the USA.
- <b>Incident_Rate</b> - cases per 100,000 persons.
- <b>People_Tested</b> - Total number of people who have been tested.
- <b>People_Hospitalized</b> - Total number of people hospitalized. (Nullified on Aug 31, see [Issue #3083](https://github.com/CSSEGISandData/COVID-19/issues/3083))
- <b>Mortality_Rate</b> - Number recorded deaths \* 100/ Number confirmed cases.
- <b>UID</b> - Unique Identifier for each row entry.
- <b>ISO3</b> - Officialy assigned country code identifiers.
- <b>Testing_Rate</b> - Total test results per 100,000 persons. The "total test results" are equal to "Total test results (Positive + Negative)" from [COVID Tracking Project](https://covidtracking.com/).
- <b>Hospitalization_Rate</b> - US Hospitalization Rate (%): = Total number hospitalized / Number cases. The "Total number hospitalized" is the "Hospitalized – Cumulative" count from [COVID Tracking Project](https://covidtracking.com/). The "hospitalization rate" and "Total number hospitalized" is only presented for those states which provide cumulative hospital data. (Nullified on Aug 31, see [Issue #3083](https://github.com/CSSEGISandData/COVID-19/issues/3083))


## Neural Network Model

For more details of Neural Network Model please refer to `neural_network.py`.

In this class we train based on Neural Network and we use GridSearch to find the best [parameters](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier)

You can add/remove parameters and their values to see how to find the optimal NN settings. Please only modify the following in  `neural_network.py`

```
self.parameters = {
    'hidden_layer_sizes': [(80, 80), (70, 70), (60, 60)],
    'activation': ['relu'],
    'solver': ['adam'],
    'learning_rate': ['adaptive'],
    'learning_rate_init': [0.0001, 0.001, 0.005, 0.0005]
} 
```
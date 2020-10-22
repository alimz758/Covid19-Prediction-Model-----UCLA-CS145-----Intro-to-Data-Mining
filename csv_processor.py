import pandas as pd
from linear_regression import LinearRegression

start_month = 4
end_month = 8


# daily_report_04_12_df = pd.read_csv(
#     "data/daily_report/04-12-2020.csv", engine="python")

# print(daily_report_04_12_df.head())


# y_cases = daily_report_04_12_df['Confirmed']
# y_deaths = daily_report_04_12_df['Deaths']

# x = daily_report_04_12_df.drop(
#     ['Province_State', 'Country_Region', 'Last_Update', 'Lat', 'Long_', 'UID', 'ISO3', 'Testing_Rate', 'Hospitalization_Rate'], axis=1)

lm = LinearRegression()
lm.load_data('./data/daily_report/04-12-2020.csv',
             './data/daily_report/04-12-2020.csv')
print('Training data shape: ', lm.train_x.shape)
print('Training labels shape:', lm.train_y.shape)

training_error = 0
testing_error = 0

lm.normalize()

beta = lm.train('0')
prediction = lm.predict(lm.train_x, beta)

print(prediction)
print(lm.train_y)


training_error = lm.compute_mse(prediction, lm.train_y)

prediction = lm.predict(lm.test_x, beta)
testing_error = lm.compute_mse(prediction, lm.test_y)

print('Training error is: ', training_error)
print('Testing error is: ', testing_error)

# print(x)

# def getDataframe(filePath):
#     dataframe = pd.read_csv(filePath)
#     y = dataframe['y']
#     x = dataframe.drop('y', axis=1)
#     return x, y

# df.drop(['B', 'C'], axis=1)

# from hw1code.linear_regression import LinearRegression

# lm=LinearRegression()
# lm.load_data('./data/linear-regression-train.csv','./data/linear-regression-test.csv')

# lm.train_x

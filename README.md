# Covid19-Prediction-Model-----UCLA-CS145-----Intro-to-Data-Mining

Course project for UCLA CS145, Introduction to Data Mining

## Initializing Input Data

### Partitioning daily report data by states

To transform input data, run:

> python transform_input.py

It will then create a csv file for each states, each containing its state's daily report. Miscellaneous states from the input data set are ignored

**NOTE** Each time this script is ran, all the `<state>.csv` files are truncated an refilled from the daily report files.

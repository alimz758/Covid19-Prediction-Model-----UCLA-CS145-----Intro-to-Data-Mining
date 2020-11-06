import pandas as pd
import os.path
import numpy as np
from create_input_df import CreateDataframe
from neural_network import NeuralNetwork


def get_state_df(self, df, target):
    for column in df:
        if column != 'Days' and column != 'Date':
            if(target == "Confirmed"):
                self.confirmed_states_df[column] = df[column]
            else:
                self.deaths_states_df[column] = df[column]


def train_df_model(self, df, target):
    for state_name, state_df in df.items():
        nn = NeuralNetwork()
        print(df[state_name])
        nn.train(self.training_days, df[state_name], self.hidden_layer_sizes,
                 self.learning_rate_init, self.max_iter, self.learning_rate)
        if (target == "Confirmed"):
            self.confirmed_states_model_list.append(nn)
        else:
            self.deaths_states_model_list.append(nn)


def get_pred(self, model_list, target):
    confirmed_pred = {}
    deaths_pred = {}
    for model in model_list:
        if (target == "Confirmed"):
            confirmed_pred[model.state_name] = model.get_predictions(self.days)
        else:
            deaths_pred[model.state_name] = model.get_predictions(self.days)
    return confirmed_pred if target == "Confirmed" else deaths_pred


class GetNNPredicitons(object):
    def __init__(self):
        df = CreateDataframe()
        self.confirmed_df = df.get_final_df('Confirmed')
        self.deaths_df = df.get_final_df('Deaths')
        self.confirmed_states_pred_dict = {}
        self.deaths_states_pred_dict = {}
        self.confirmed_states_model_list = []
        self.deaths_states_model_list = []
        self.confirmed_states_df = {}
        self.deaths_states_df = {}
        self.training_days = np.array(self.confirmed_df["Days"]).reshape(-1, 1)
        self.learning_rate_init = 0.0001  # 0.00001  made it worse
        self.max_iter = 500000
        self.hidden_layer_sizes = (80, 80)
        self.learning_rate = "adaptive"  # default 'constant'
        self.period = len(self.training_days)
        self.upcoming_days = 26
        self.days = np.array(
            range(self.period, self.period + self.upcoming_days)).reshape(-1, 1)
        # Create a dictionary of DF
        get_state_df(self, self.confirmed_df, "Confirmed")
        get_state_df(self, self.deaths_states_df, "Deaths")
        train_df_model(self, self.confirmed_states_df, "Confirmed")
        train_df_model(self, self.deaths_states_df, "Deaths")
        # NN predicitions, store prediciton fo each state in a dict
        self.confirmed_states_pred_dict = get_pred(
            self, self.confirmed_states_model_list, "Confirmed")
        self.deaths_states_pred_dict = get_pred(
            self, self.deaths_states_model_list, "Deaths")

    def get_nn_pred(self, state_name, target_name):
        if (target_name == "Confirmed"):
            return self.confirmed_states_pred_dict[state_name]
        elif (target_name == "Deaths"):
            return self.deaths_states_pred_dict[state_name]
        else:
            raise ValueError("Pass in Confirmed or Deaths")


def main():
    x = GetNNPredicitons()
    x.get_nn_pred("Alabama", "Confirmed")


main()

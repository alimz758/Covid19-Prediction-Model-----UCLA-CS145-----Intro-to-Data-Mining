from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np


from ..utils.prediction_model import PredictionModel
from ..utils.create_input_df import CreateDataframe


class NeuralNetwork(PredictionModel):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.mlp = MLPRegressor(max_iter=500000)
        self.parameters = {
            'hidden_layer_sizes': [(80, 80), (70, 70), (60, 60)],
            'activation': ['relu'],
            'solver': ['adam'],
            'learning_rate': ['adaptive'],
            'learning_rate_init': [0.0001, 0.001, 0.005, 0.0005]
        }
        self.clf = None

    def train(self, predict_state, predict_field):
        self.clf = GridSearchCV(self.mlp,  self.parameters, n_jobs=-1, cv=3)
        train_df = self.assign_train_df(predict_field)
        self.clf.fit(self.x_train, train_df[predict_state])
        print('Optimal NN settings for {} are {}:\n'.format(predict_state, self.clf.best_params_))

    def predict(self):
        return np.round(self.clf.predict(self.x_test), 0).astype(np.int32)


def main():
    poly = NeuralNetwork()
    poly.train("Alabama", "Confirmed")
    prediction = poly.predict()
    print(prediction)


if __name__ == "__main__":
    main()

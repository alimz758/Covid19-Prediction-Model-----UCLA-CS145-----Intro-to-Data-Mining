from sklearn.neural_network import MLPRegressor
import numpy as np


from prediction_model import PredictionModel
from create_input_df import CreateDataframe


class NeuralNetwork(PredictionModel):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.model = MLPRegressor()

    def train(self, predict_state, predict_field, solver="adam", activation="relu",  hidden_layer_sizes=[60, 60],
              learning_rate_init=0.001, max_iter=2000, learning_rate='constant'):
        self.model = MLPRegressor(solver=solver, activation=activation,  hidden_layer_sizes=hidden_layer_sizes,
                                  learning_rate_init=learning_rate_init, max_iter=max_iter, learning_rate=learning_rate)

        train_df = self.assign_train_df(predict_field)
        self.model.fit(self.x_train, train_df[predict_state])

    def predict(self):
        return np.round(self.model.predict(self.x_test), 0).astype(np.int32)


def main():
    poly = NeuralNetwork()
    poly.train("Alabama", "Confirmed")
    prediction = poly.predict()
    print(prediction)


if __name__ == "__main__":
    main()

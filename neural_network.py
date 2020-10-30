from sklearn.neural_network import MLPRegressor
import numpy as np

class NeuralNetwork(object):
    
    def __init__(self, state_name):
        self.state_name = state_name
        self.model = None

    def train(self, time_period, state, hidden_layer_sizes = [60, 60], learning_rate_init = 0.001, max_iter = 2000, learning_rate = 'constant'):
        #‘adam’ refers to a stochastic gradient-based optimizer proposed by Kingma, Diederik, and Jimmy Ba
        #‘relu’, the rectified linear unit function, returns f(x) = max(0, x)
        # check https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier for more details
        
        self.model = MLPRegressor(solver="adam", activation="relu",  hidden_layer_sizes = hidden_layer_sizes,
                                    learning_rate_init = learning_rate_init, max_iter = max_iter, learning_rate = learning_rate)
        self.model.fit(time_period, state)

    def get_predictions(self, time_period):
        return np.round(self.model.predict(time_period), 0).astype(np.int32)
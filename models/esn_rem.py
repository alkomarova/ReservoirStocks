import math

import numpy as np
from hyperopt import fmin, hp
from reservoirpy.nodes import Reservoir, Ridge, Input
from sklearn.metrics import mean_absolute_error


class ESN2Predictions:

    def __init__(self, data, window_size=100, forecast_size=1, test_size=0.3, model_type='simple') -> None:
        self.data = data
        self.window_size = window_size
        self.forecast_size = forecast_size
        self.test_size = test_size
        self.model_type = model_type

        self.data_train_len = round(self.data.shape[0] * (1 - self.test_size))
        self.data_test_len = self.data.shape[0] - self.data_train_len
        self.X_train = data[:self.data_train_len]
        self.Y_train = data[1:self.data_train_len + 1]
        self.X_test = data[self.data_train_len:-1]
        self.Y_test = data[self.data_train_len + 1:]

        self.feature_len = self.X_test.shape[1]

    def create_esn_model(self, params):

        if self.model_type == 'simple':
            reservoir = Reservoir(100, iss=1, lr=params["lr"], sr=params["sr"])
            ridge = Ridge(self.feature_len, ridge=1e-7)
            model = reservoir >> ridge
            print(self.X_train, self.Y_train)
            model = model.fit(self.X_train, self.Y_train, warmup=10)

        elif self.model_type == 'hierarchical':
            reservoir1 = Reservoir(100, lr=params["lr"], sr=params["sr"])
            reservoir2 = Reservoir(100, lr=params["lr"], sr=params["sr"])

            readout1 = Ridge(self.feature_len, ridge=1e-5, name="readout1-1")
            readout2 = Ridge(self.feature_len, ridge=1e-5, name="readout2-1")

            model = reservoir1 >> readout1 >> reservoir2 >> readout2
            model = model.fit(self.X_train, {"readout1-1": self.Y_train, "readout2-1": self.Y_train})
        elif self.model_type == 'deep':
            data = Input()
            reservoir1 = Reservoir(100, lr=params["lr"], sr=params["sr"])
            reservoir2 = Reservoir(100, lr=params["lr"], sr=params["sr"])
            reservoir3 = Reservoir(100, lr=params["lr"], sr=params["sr"])

            readout = Ridge(self.feature_len, ridge=1e-5)

            model = reservoir1 >> reservoir2 >> reservoir3 & \
                    data >> [reservoir1, reservoir2, reservoir3] >> readout
            model = model.fit(self.X_train, self.Y_train, warmup=10)
        return model

    def objective(self, params):
        model = self.create_esn_model(params)
        model.fit(self.X_train, self.Y_train, warmup=10)
        X_train_temp = self.X_train
        X_test_temp = self.X_test
        test_blocks = math.floor(self.data_test_len / self.forecast_size)
        Y_pred_test = []

        for i in range(test_blocks):
            warmup_y = model.run(X_train_temp[-self.window_size:], reset=True)
            Y_pred = np.empty((self.forecast_size, self.feature_len))
            x = warmup_y[-1].reshape(1, -1)

            for i in range(self.forecast_size):
                x = model(x)
                Y_pred[i] = x

            Y_pred = Y_pred[:, 0]
            Y_pred_test = np.append(Y_pred_test, Y_pred)
            X_train_temp = np.vstack([X_train_temp, X_test_temp[:][:self.forecast_size]])
            X_test_temp = X_test_temp[:][self.forecast_size:]

        mae = mean_absolute_error(self.Y_test[:, 0][:len(Y_pred_test)], Y_pred_test)
        return mae

    def get_optimal_parameters(self):
        space = {
            'sr': hp.loguniform('sr', 1e-6, 10),
            'lr': hp.loguniform('lr', 1e-3, 0.9),
        }
        best = fmin(fn=self.objective, space=space, max_evals=10)
        return best

    def get_predictions(self, opt=True):
        if opt == True:
            params = self.get_optimal_parameters()
        else:
            params = {'lr': 0.3, 'sr': 0.125}
        print(params)
        esn_model = self.create_esn_model(params)
        X_train_temp = self.X_train
        X_test_temp = self.X_test
        test_blocks = math.floor(self.data_test_len / self.forecast_size)

        Y_pred_test = []

        for i in range(test_blocks):

            warmup_y = esn_model.run(X_train_temp[-self.window_size:], reset=True)
            Y_pred = np.empty((self.forecast_size, self.feature_len))
            x = warmup_y[-1].reshape(1, -1)

            for i in range(self.forecast_size):
                x = esn_model(x)
                Y_pred[i] = x

            Y_pred = Y_pred[:, 0]
            Y_pred_test = np.append(Y_pred_test, Y_pred)
            X_train_temp = np.vstack([X_train_temp, X_test_temp[:][:self.forecast_size]])
            X_test_temp = X_test_temp[:][self.forecast_size:]

        return Y_pred_test

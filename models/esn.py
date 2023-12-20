import numpy as np
import json
from reservoirpy.nodes import Reservoir, Ridge, Input
from reservoirpy.observables import nrmse, rsquare
from sklearn import metrics
from reservoirpy.hyper import research
import math
import hyperopt


class ESNPredictions:

    def __init__(self, data, window_size = 100, forecast_size = 1, test_size = 0.3, model_type = 'simple') -> None:
        self.data = data
        self.window_size = window_size
        self.forecast_size = forecast_size
        self.test_size= test_size
        self.model_type = model_type

        self.data_train_len = round(self.data.shape[0]*(1-self.test_size))
        self.data_test_len = self.data.shape[0] - self.data_train_len
        self.X_train = data[:self.data_train_len]
        self.Y_train = data[1:self.data_train_len+1]
        self.X_test = data[self.data_train_len:-1]
        self.Y_test = data[self.data_train_len+1:]

        self.feature_len = self.X_test.shape[1]

    def objective(self, dataset, config, *, iss, N, sr, lr, ridge, seed):
        train_data, validation_data = dataset
        X_train, y_train = train_data
        X_test, y_test = validation_data

        instances = config["instances_per_trial"]

        variable_seed = seed

        losses = [];
        r2s = [];
        for n in range(instances):
            # Build your model given the input parameters
            reservoir = Reservoir(N,
                                  sr=sr,
                                  lr=lr,
                                  inut_scaling=iss,
                                  seed=variable_seed)

            readout = Ridge(self.feature_len, ridge=ridge)

            model = reservoir >> readout

            # Train your model and test your model.
            predictions = model.fit(X_train, y_train) \
                .run(X_test)

            loss = metrics.mean_absolute_error(y_test, predictions) #, norm_value=np.ptp(X_train))
            r2 = rsquare(y_test, predictions)

            # Change the seed between instances
            variable_seed += 1

            losses.append(loss)
            r2s.append(r2)

        # Return a dictionnary of metrics. The 'loss' key is mandatory when
        # using hyperopt.
        return {'loss': np.mean(losses),
                'r2': np.mean(r2s)}

    def get_optimal_parameters(self):

        dataset = ((self.X_train, self.Y_train), (self.X_test, self.Y_test))
        hyperopt_config = {
            "exp": f"hyperopt-multiscroll",  # the experimentation name
            "hp_max_evals": 50,  # the number of different sets of parameters hyperopt has to try
            "hp_method": "random",  # the method used by hyperopt to choose those sets (see below)
            "seed": 42,  # the random state seed, to ensure reproducibility
            "instances_per_trial": 3,  # how many random ESN will be tried with each sets of parameters
            "hp_space": {  # what are the ranges of parameters explored
                "N": ["choice", 100],  # the number of neurons is fixed to 300
                "sr": ["loguniform", 1e-3, 10],  # the spectral radius is log-uniformly distributed between 1e-6 and 10
                "lr": ["loguniform", 1e-6, 1],  # idem with the leaking rate, from 1e-3 to 1
                "iss": ["choice", 0.5],  # the input scaling is fixed
                "ridge": ["choice", 1e-7],  # and so is the regularization parameter.
                "seed": ["choice", 1234]  # an other random seed for the ESN initialization
            }
        }

        # we precautionously save the configuration in a JSON file
        # each file will begin with a number corresponding to the current experimentation run number.
        with open(f"{hyperopt_config['exp']}.config.json", "w+") as f:
            json.dump(hyperopt_config, f)
        best = research(self.objective, dataset, f"{hyperopt_config['exp']}.config.json", ".")
        print(best)
        return best[0]['lr'], best[0]['sr']

    def get_model(self, opt = True):
        if opt == True:
            opt_lr, opt_sr = self.get_optimal_parameters()
        else:
            opt_lr, opt_sr = 0.3 , 0.125

        if self.model_type == 'simple':
            reservoir = Reservoir(100, iss=1, lr=opt_lr, sr=opt_sr)
            ridge = Ridge(self.feature_len, ridge=1e-7)
            model = reservoir >> ridge
            model = model.fit(self.X_train, self.Y_train, warmup=10)
        elif self.model_type == 'hierarchical':
            reservoir1 = Reservoir(100, lr=opt_lr, sr=opt_sr)
            reservoir2 = Reservoir(100, lr=opt_lr, sr=opt_sr)

            readout1 = Ridge(self.feature_len, ridge=1e-7)
            readout2 = Ridge(self.feature_len, ridge=1e-7)

            model = reservoir1 >> readout1 >> reservoir2 >> readout2
            model = model.fit(self.X_train, self.Y_train)
        elif self.model_type == 'deep':
            data = Input()
            reservoir1 = Reservoir(100, lr=opt_lr, sr=opt_sr)
            reservoir2 = Reservoir(100, lr=opt_lr, sr=opt_sr)
            reservoir3 = Reservoir(100, lr=opt_lr, sr=opt_sr)

            readout = Ridge(self.feature_len, ridge=1e-7)

            model = reservoir1 >> reservoir2 >> reservoir3 & \
                    data >> [reservoir1, reservoir2, reservoir3] >> readout
            model = model.fit(self.X_train, self.Y_train, warmup=10)
        return model

    def get_predictions(self, opt = True):

        esn_model = self.get_model(opt)
        X_train_temp = self.X_train
        X_test_temp = self.X_test
        test_blocks = math.floor(self.data_test_len/self.forecast_size)
        #print(test_blocks, self.data_test_len, self.forecast_size)
        Y_pred_test = []

        for i in range(test_blocks):

            #print('Тренируемся на:', X_train_temp[:-self.window_size][:, 0])
            warmup_y = esn_model.run(X_train_temp[-self.window_size:], reset=True)
            Y_pred = np.empty((self.forecast_size, self.feature_len))
            x = warmup_y[-1].reshape(1, -1)

            for i in range(self.forecast_size):
                x = esn_model(x)
                Y_pred[i] = x

            Y_pred = Y_pred[:, 0]
            #print('Спрогнозировали:', Y_pred)
            Y_pred_test = np.append(Y_pred_test, Y_pred)
            X_train_temp = np.vstack([X_train_temp, X_test_temp[:][:self.forecast_size]])
            X_test_temp = X_test_temp[:][self.forecast_size:]

        return Y_pred_test





